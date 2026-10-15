// SPDX-License-Identifier: Apache-2.0
//
// The light pass: per view culling, the stencil volume, and the shade.
//
// A spot light is shaded through a capped cone covering the volume it can
// reach, 18 sides by 12 slices. The cone is drawn with the engine's own radial
// light vertex shader, DeferredLightVertexShaders.usf under SHADER_RADIAL_LIGHT,
// which generates the vertex positions from SV_VertexID and the cone
// parameters. Only the topology is supplied here.
//
// The culling and the fade match what the engine applies to its own local
// lights, so a light from this plugin and an engine light of the same size
// switch off and fade out at the same distance. That is what makes the two
// comparable side by side.
//
//   Stencil    the volume tags stencil only, then a fullscreen triangle shades
//              the tagged pixels. See AddStencilTagPass.
//   Blend      additive into RGBA. Alpha carries the non specular luminance the
//              subsurface profile pass reads, as it does for engine lights.

#include "SWRTSpotLightRendering.h"

#include "ScenePrivate.h"
#include "SceneRendering.h"
#include "LightRendering.h"
#include "SceneTextureParameters.h"
#include "SystemTextures.h"
#include "PipelineStateCache.h"
#include "RenderUtils.h"
#include "GlobalShader.h"
#include "ShaderParameterStruct.h"
#include "DataDrivenShaderPlatformInfo.h"
#include "RHIResourceUtils.h"
#include "RHIStaticStates.h"
#include "PixelShaderUtils.h"
// STENCIL_SANDBOX_MASK, the stencil bit the tagged lighting path borrows.
#include "PostProcess/SceneRenderTargets.h"

/*=============================================================================
	THE ONE DEBUG SWITCH

	Every view here needs the trace's per thread counters, which exist only in
	the stats permutation, so one cvar turns the counters, the log and the view
	on together. Nonzero also compiles the atomics in, so nothing here is a
	timing configuration.
=============================================================================*/
static TAutoConsoleVariable<int32> GSWRTDebug(
	TEXT("r.SWRTSpotLight.Debug"),
	0,
	TEXT("Stats log and debug views, 0 to 4. Nonzero logs the traversal counts and costs atomics.\n")
	TEXT("  0  off (default), compiled out entirely.\n")
	TEXT("  1  shadow term: green lit, red shadowed.\n")
	TEXT("  2  lane waste: this lane's work over what its warp waited for. Dark is wasted silicon;\n")
	TEXT("     scattered dark means ray lengths differ and only reordering helps, solid dark blocks\n")
	TEXT("     mean whole regions idle together and can be dispatched away.\n")
	TEXT("  3  ray iteration heatmap: node visits a ray, over r.SWRTSpotLight.Trace.HeatmapRange.\n")
	TEXT("     Counts both the top level and the BLAS walk.\n")
	TEXT("  4  warp mixed: white where a warp's live lanes disagreed about hit and miss. White is\n")
	TEXT("     the only part a ray sort could improve."),
	ECVF_RenderThreadSafe);

int32 SWRTSpot::GetDebugMode()
{
	return FMath::Clamp(GSWRTDebug.GetValueOnRenderThread(), 0, 4);
}

/**
 * Which measurement the trace writes into the mask instead of a shadow term.
 *
 * Mode 1 false colours the real shadow term, so it asks for no measurement.
 * Everything above it shares one numbering with the shader and the light
 * shader, so a view has the same number everywhere.
 */
uint32 SWRTSpot::GetTraceView()
{
	const int32 Mode = SWRTSpot::GetDebugMode();
	return Mode >= 2 ? (uint32)Mode : 0u;
}

/*=============================================================================
	STENCIL VOLUME GEOMETRY

	The stencil volume is a capped cone. Its vertices are generated in the
	vertex shader from SV_VertexID and the cone parameters, so what is needed
	here is only the topology: which vertices form which triangles.
=============================================================================*/
namespace
{
	/**
	 * Index buffer for the stencil cone.
	 *
	 * The vertices form two rings of the same shape, laid out one after the
	 * other: the cone body first, then the spherical cap that closes it. Each
	 * ring is a grid of NumSlices rows by NumSides columns, wrapping in the
	 * column direction, and each cell of that grid becomes two triangles.
	 *
	 * The cap is wound the other way round because it faces outward from the
	 * far end, and the stencil pass needs both rings to agree about which side
	 * is the front.
	 */
	class FSWRTStencilConeIndexBuffer : public FIndexBuffer
	{
	public:
		/** Columns: a side runs from the cone's origin out to its rim. */
		static const int32 NumSides = 18;
		/** Rows: a slice is one ring of vertices around the cone's axis. */
		static const int32 NumSlices = 12;

		static const uint32 NumVerts = NumSides * NumSlices * 2;

		virtual void InitRHI(FRHICommandListBase& RHICmdList) override
		{
			TArray<uint16> Indices;
			Indices.Reserve((NumSlices - 1) * NumSides * 12);

			// One grid cell, as two triangles. bOutward picks the winding.
			auto AddCell = [&Indices](int32 TopLeft, int32 TopRight, int32 BottomLeft, int32 BottomRight, bool bOutward)
			{
				if (bOutward)
				{
					Indices.Add((uint16)TopLeft);
					Indices.Add((uint16)BottomLeft);
					Indices.Add((uint16)TopRight);

					Indices.Add((uint16)TopRight);
					Indices.Add((uint16)BottomLeft);
					Indices.Add((uint16)BottomRight);
				}
				else
				{
					Indices.Add((uint16)TopLeft);
					Indices.Add((uint16)TopRight);
					Indices.Add((uint16)BottomLeft);

					Indices.Add((uint16)BottomLeft);
					Indices.Add((uint16)TopRight);
					Indices.Add((uint16)BottomRight);
				}
			};

			// Both rings, in the order the vertex shader lays them out.
			auto AddRing = [&AddCell](int32 FirstVertex, bool bOutward)
			{
				for (int32 Row = 0; Row + 1 < NumSlices; ++Row)
				{
					const int32 ThisRow = FirstVertex + Row * NumSides;
					const int32 NextRow = ThisRow + NumSides;

					for (int32 Column = 0; Column < NumSides; ++Column)
					{
						// Wraps, so the last column closes back onto the first.
						const int32 NextColumn = (Column + 1) % NumSides;

						AddCell(
							ThisRow + Column, ThisRow + NextColumn,
							NextRow + Column, NextRow + NextColumn,
							bOutward);
					}
				}
			};

			AddRing(0, false);
			AddRing(NumSides * NumSlices, true);

			NumIndices = Indices.Num();

			IndexBufferRHI = UE::RHIResourceUtils::CreateIndexBufferFromArray(
				RHICmdList, TEXT("FSWRTStencilConeIndexBuffer"), EBufferUsageFlags::Static, MakeConstArrayView(Indices));
		}

		int32 GetIndexCount() const { return NumIndices; }

	protected:
		int32 NumIndices = 0;
	};

	/** Zeroed: the vertex shader generates positions from SV_VertexID. */
	class FSWRTStencilConeVertexBuffer : public FVertexBuffer
	{
	public:
		static const int32 NumVerts = FSWRTStencilConeIndexBuffer::NumSides * FSWRTStencilConeIndexBuffer::NumSlices * 2;

		virtual void InitRHI(FRHICommandListBase& RHICmdList) override
		{
			TArray<FVector4f> Verts;
			Verts.Empty(NumVerts);
			for (int32 s = 0; s < NumVerts; s++)
			{
				Verts.Emplace(0, 0, 0, 0);
			}

			VertexBufferRHI = UE::RHIResourceUtils::CreateVertexBufferFromArray(
				RHICmdList, TEXT("FSWRTStencilConeVertexBuffer"), EBufferUsageFlags::Static, MakeConstArrayView(Verts));
		}

		int32 GetVertexCount() const { return NumVerts; }
	};

	TGlobalResource<FSWRTStencilConeIndexBuffer> GSWRTStencilConeIndexBuffer;
	TGlobalResource<FSWRTStencilConeVertexBuffer> GSWRTStencilConeVertexBuffer;
}

/*=============================================================================
	SHADERS
=============================================================================*/

/**
 * The engine's radial light vertex shader, compiled as a plugin type because
 * FDeferredLightVS is not exported. Same file, same entry point, same define.
 */
class FSWRTLightVS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FSWRTLightVS);
	SHADER_USE_PARAMETER_STRUCT(FSWRTLightVS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
		SHADER_PARAMETER(FVector4f, StencilingGeometryPosAndScale)
		SHADER_PARAMETER(FVector4f, StencilingConeParameters)
		SHADER_PARAMETER(FMatrix44f, StencilingConeTransform)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("SHADER_RADIAL_LIGHT"), 1);
	}
};

IMPLEMENT_GLOBAL_SHADER(FSWRTLightVS, "/Engine/Private/DeferredLightVertexShaders.usf", "VertexMain", SF_Vertex);

/**
 * The shade. Its BRDF is the engine's GetDynamicLighting and its shadow term is
 * one traced ray, read straight out of the mask.
 */
class FSWRTSpotLightPS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FSWRTSpotLightPS);
	SHADER_USE_PARAMETER_STRUCT(FSWRTSpotLightPS, FGlobalShader);

	/** Off when nothing could be traced, which shades the light unshadowed. */
	class FShadowed : SHADER_PERMUTATION_BOOL("SWRT_SHADOWED");
	using FPermutationDomain = TShaderPermutationDomain<FShadowed>;

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
		SHADER_PARAMETER_STRUCT_INCLUDE(FSceneTextureShaderParameters, SceneTextures)

		SHADER_PARAMETER(FVector3f, SWRTLightTranslatedWorldPosition)
		SHADER_PARAMETER(float, SWRTLightInvRadius)
		SHADER_PARAMETER(FVector3f, SWRTLightColor)
		SHADER_PARAMETER(float, SWRTLightFalloffExponent)
		SHADER_PARAMETER(FVector3f, SWRTLightDirection)
		SHADER_PARAMETER(FVector3f, SWRTLightTangent)
		SHADER_PARAMETER(FVector2f, SWRTLightSpotAngles)
		SHADER_PARAMETER(float, SWRTLightSourceRadius)
		SHADER_PARAMETER(float, SWRTLightSoftSourceRadius)
		SHADER_PARAMETER(float, SWRTLightSourceLength)
		SHADER_PARAMETER(float, SWRTLightSpecularScale)
		SHADER_PARAMETER(float, SWRTLightDiffuseScale)
		SHADER_PARAMETER(float, SWRTContactShadowLength)
		SHADER_PARAMETER(float, SWRTContactShadowCastingIntensity)
		SHADER_PARAMETER(float, SWRTContactShadowNonCastingIntensity)
		SHADER_PARAMETER(uint32, SWRTShadowedBits)

		/** 0 shadowed, 1 lit, one texel a pixel. */
		SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float>, SWRTShadowMask)

		/** r.SWRTSpotLight.Debug. */
		SHADER_PARAMETER(uint32, SWRTDebugMode)

		RENDER_TARGET_BINDING_SLOTS()
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5)
			&& !Substrate::IsSubstrateEnabled();
	}
};

IMPLEMENT_GLOBAL_SHADER(FSWRTSpotLightPS, "/Plugins/PrototypeSWRTShadows/SWRTSpotLightPS.usf", "MainPS", SF_Pixel);

namespace
{
	/**
	 * A pass that only draws a light volume still has to declare View.
	 * FViewUniformShaderParameters has a static binding slot, and RDG binds
	 * static slot uniform buffers per pass from the pass parameter struct, so a
	 * pass that omits it leaves the slot unbound, the vertex shader reads a
	 * zeroed TranslatedWorldToClip, and the draw silently produces no pixels.
	 */
	BEGIN_SHADER_PARAMETER_STRUCT(FSWRTVolumePassParameters, )
		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
		RENDER_TARGET_BINDING_SLOTS()
	END_SHADER_PARAMETER_STRUCT()

	float GetEngineCVarFloat(const TCHAR* Name, float Default)
	{
		const IConsoleVariable* Variable = IConsoleManager::Get().FindConsoleVariable(Name);
		return Variable != nullptr ? Variable->GetFloat() : Default;
	}

	/*=========================================================================
		VISIBILITY
	=========================================================================*/

	/**
	 * How small this light's bounding sphere is on screen, squared and scaled.
	 *
	 * One term decides both whether the light is worth drawing and how far it
	 * has faded by size, so it is solved once here. Multiply it by a squared
	 * distance: at 1 the sphere has shrunk to the threshold.
	 */
	float ComputeScreenSizeTerm(const FViewInfo& View, float BoundingRadius)
	{
		const float MinScreenRadiusForLights = GetEngineCVarFloat(TEXT("r.MinScreenRadiusForLights"), 0.03f);

		// The cap keeps a very small light from being culled instantly.
		const float ScreenRadius = FMath::Min(0.0002f, MinScreenRadiusForLights / BoundingRadius);

		return FMath::Square(ScreenRadius * View.LODDistanceFactor);
	}

	/**
	 * Whether this light is drawn for this view at all, and how much it has
	 * faded if it is. Matches what the engine does for its own local lights, so
	 * a plugin light and an engine light switch off at the same place.
	 *
	 * OutFadeFactor multiplies the light colour. Zero means do not draw.
	 */
	bool IsSWRTSpotLightVisibleInView(const FViewInfo& View, const FSWRTSpotLightRenderData& Light, float& OutFadeFactor)
	{
		OutFadeFactor = 1.0f;

		const FEngineShowFlags& ShowFlags = View.Family->EngineShowFlags;
		if (!ShowFlags.Lighting || !ShowFlags.DirectLighting || !ShowFlags.SpotLights)
		{
			return false;
		}

		const FSphere& BoundingSphere = Light.Bounds;
		if (!View.GetCullingFrustum().IntersectSphere(BoundingSphere.Center, BoundingSphere.W))
		{
			return false;
		}

		// An orthographic view has no distance falloff to apply.
		if (!View.IsPerspectiveProjection())
		{
			return true;
		}

		const float LightMaxDrawDistanceScale = GetEngineCVarFloat(TEXT("r.LightMaxDrawDistanceScale"), 1.0f);
		const float ScaledMaxDistance = Light.MaxDrawDistance * LightMaxDrawDistanceScale;
		const float ScreenSizeTerm = ComputeScreenSizeTerm(View, BoundingSphere.W);

		const bool bHasDrawDistance = Light.MaxDrawDistance > 0.0f;

		// ---- Culled, measured from where culling is done from
		{
			const float CullDistanceSquared = (BoundingSphere.Center - View.CullingOrigin).SizeSquared();

			if (ScreenSizeTerm * CullDistanceSquared >= 1.0f)
			{
				return false;
			}

			if (bHasDrawDistance && CullDistanceSquared >= FMath::Square(ScaledMaxDistance))
			{
				return false;
			}
		}

		// ---- Faded, measured from the eye
		{
			const float FadeDistanceSquared =
				(BoundingSphere.Center - View.ViewMatrices.GetViewOrigin()).SizeSquared();

			// Shrinks to nothing over the last sixth of the way to the cull.
			const float SizeFade =
				FMath::Clamp(6.0f - 6.0f * (ScreenSizeTerm * FadeDistanceSquared), 0.0f, 1.0f);

			const float FadeRange = FMath::Max(0.001f, Light.MaxDistanceFadeRange * LightMaxDrawDistanceScale);
			const float DistanceFade = bHasDrawDistance
				? FMath::Clamp((ScaledMaxDistance - FMath::Sqrt(FadeDistanceSquared)) / FadeRange, 0.0f, 1.0f)
				: 1.0f;

			OutFadeFactor = SizeFade * DistanceFade;
		}

		return OutFadeFactor > 0.0f;
	}

	/**
	 * The device depth range the light's bounding sphere spans, for the depth
	 * bounds test. Taking the two points on the view axis is enough: the sphere
	 * cannot reach past either of them in depth.
	 *
	 * A point behind the eye projects through infinity, so each end falls back
	 * to its own extreme of the reversed Z range rather than to a wrong number.
	 */
	void CalculateLightNearFarDepthFromBounds(const FViewInfo& View, const FSphere& LightBounds, float& NearDepth, float& FarDepth)
	{
		const FMatrix ViewProjection = View.ViewMatrices.GetViewProjectionMatrix();
		const FVector ViewDirection = View.GetViewDirection();

		// Device depth of a point, plus whether it was in front of the eye.
		auto ProjectDepth = [&ViewProjection](const FVector& Point, bool& bOutInFront)
		{
			const FVector4 Clip = ViewProjection.TransformFVector4(FVector4(Point, 1.0));
			bOutInFront = Clip.W >= 0.0;
			return (float)(Clip.Z / Clip.W);
		};

		bool bNearInFront = false;
		bool bFarInFront = false;

		const float NearCandidate =
			ProjectDepth(LightBounds.Center - LightBounds.W * ViewDirection, bNearInFront);
		const float FarCandidate =
			ProjectDepth(LightBounds.Center + LightBounds.W * ViewDirection, bFarInFront);

		// Reversed Z, so the near end is the larger value.
		NearDepth = bNearInFront ? NearCandidate : 1.0f;
		FarDepth = bFarInFront ? FarCandidate : 0.0f;
	}

	/*=========================================================================
		LIGHT VOLUME
	=========================================================================*/
	struct FSWRTLightVolume
	{
		FSWRTLightVS::FParameters VSParameters;
		bool bUseDepthBounds = false;
		float NearDepth = 1.0f;
		float FarDepth = 0.0f;
	};

	FSWRTLightVolume SetupLightVolume(const FViewInfo& View, const FSWRTSpotLightRenderData& Light)
	{
		FSWRTLightVolume Volume;

		const FVector PreViewTranslation = View.ViewMatrices.GetPreViewTranslation();

		// What the radial light vertex shader needs to build the cone.
		FSWRTLightVS::FParameters& VSParameters = Volume.VSParameters;
		VSParameters.View = View.ViewUniformBuffer;
		VSParameters.StencilingGeometryPosAndScale = FVector4f(0.0f, 0.0f, 0.0f, 0.0f);
		VSParameters.StencilingConeTransform = FMatrix44f(Light.LightToWorld * FTranslationMatrix(PreViewTranslation));
		VSParameters.StencilingConeParameters = FVector4f(
			FSWRTStencilConeIndexBuffer::NumSides,
			FSWRTStencilConeIndexBuffer::NumSlices,
			Light.OuterConeAngle,
			Light.Radius);

		const FSphere& LightBounds = Light.Bounds;

		Volume.bUseDepthBounds = GSupportsDepthBoundsTest && Light.bAllowDepthBoundsTest;
		if (Volume.bUseDepthBounds)
		{
			CalculateLightNearFarDepthFromBounds(View, LightBounds, Volume.NearDepth, Volume.FarDepth);
			if (Volume.NearDepth <= Volume.FarDepth)
			{
				Volume.NearDepth = 1.0f;
				Volume.FarDepth = 0.0f;
			}
		}

		return Volume;
	}

	/** Draws the cone. The vertex buffer is zeroed; the shader makes positions. */
	void DrawLightVolume(FRHICommandList& RHICmdList)
	{
		RHICmdList.SetStreamSource(0, GSWRTStencilConeVertexBuffer.VertexBufferRHI, 0);
		RHICmdList.DrawIndexedPrimitive(
			GSWRTStencilConeIndexBuffer.IndexBufferRHI, 0, 0,
			FSWRTStencilConeIndexBuffer::NumVerts, 0,
			GSWRTStencilConeIndexBuffer.GetIndexCount() / 3, 1);
	}

	/**
	 * Pass 1 of the stencil tagged path: the volume writes stencil only and tags
	 * every pixel whose surface lies inside it.
	 *
	 * The tag is the engine's stencil sandbox bit, which the layout comment in
	 * Renderer/Private/PostProcess/SceneRenderTargets.h reserves for exactly
	 * this as long as it is reset to 0 afterwards. Bits 2 to 7 still hold base
	 * pass data that later engine passes read, which is why every stencil state
	 * here masks reads and writes to the sandbox bit.
	 *
	 * Tagging uses parity rather than counting, because one bit cannot count.
	 * Both faces are drawn with the scene depth test, and every fragment that
	 * FAILS it inverts the bit:
	 *
	 *   surface in front of the volume  front and back both fail  0
	 *   surface inside the volume       only the back face fails  1
	 *   surface behind the volume       neither fails             0
	 *
	 * This is the z fail formulation, so it needs no special case when the
	 * camera is inside the volume: the clipped front faces never contribute and
	 * the back face alone decides. Reversed Z uses an infinite far plane, so
	 * back faces are never far clipped.
	 */
	void AddStencilTagPass(
		FRDGBuilder& GraphBuilder,
		const FViewInfo& View,
		FRDGTextureRef SceneDepthTexture,
		const FSWRTLightVolume& Volume,
		const TShaderMapRef<FSWRTLightVS>& VertexShader,
		const FIntRect& ViewRect)
	{
		FSWRTVolumePassParameters* TagParameters = GraphBuilder.AllocParameters<FSWRTVolumePassParameters>();
		TagParameters->View = View.ViewUniformBuffer;
		TagParameters->RenderTargets.DepthStencil = FDepthStencilBinding(
			SceneDepthTexture,
			ERenderTargetLoadAction::ELoad,
			ERenderTargetLoadAction::ELoad,
			FExclusiveDepthStencil::DepthRead_StencilWrite);

		GraphBuilder.AddPass(
			RDG_EVENT_NAME("SWRTSpotLight.StencilTag"),
			TagParameters,
			ERDGPassFlags::Raster,
			[Volume, VertexShader, ViewRect](FRDGAsyncTask, FRHICommandList& RHICmdList)
			{
				FGraphicsPipelineStateInitializer GraphicsPSOInit;
				RHICmdList.ApplyCachedRenderTargets(GraphicsPSOInit);

				GraphicsPSOInit.PrimitiveType = PT_TriangleList;
				GraphicsPSOInit.BlendState = TStaticBlendState<CW_NONE>::GetRHI();
				GraphicsPSOInit.RasterizerState = TStaticRasterizerState<FM_Solid, CM_None>::GetRHI();
				GraphicsPSOInit.DepthStencilState = TStaticDepthStencilState<
					false, CF_DepthNearOrEqual,
					true, CF_Always, SO_Keep, SO_Invert, SO_Keep,
					true, CF_Always, SO_Keep, SO_Invert, SO_Keep,
					STENCIL_SANDBOX_MASK, STENCIL_SANDBOX_MASK>::GetRHI();
				GraphicsPSOInit.bDepthBounds = Volume.bUseDepthBounds;

				GraphicsPSOInit.BoundShaderState.VertexDeclarationRHI = GetVertexDeclarationFVector4();
				GraphicsPSOInit.BoundShaderState.VertexShaderRHI = VertexShader.GetVertexShader();
				// No pixel shader: the stencil operations are the whole output.
				GraphicsPSOInit.BoundShaderState.PixelShaderRHI = nullptr;

				RHICmdList.SetViewport((float)ViewRect.Min.X, (float)ViewRect.Min.Y, 0.0f, (float)ViewRect.Max.X, (float)ViewRect.Max.Y, 1.0f);
				SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit, 0);
				SetShaderParameters(RHICmdList, VertexShader, VertexShader.GetVertexShader(), Volume.VSParameters);

				if (Volume.bUseDepthBounds)
				{
					// Reversed depth, so far is less than near.
					RHICmdList.SetDepthBounds(Volume.FarDepth, Volume.NearDepth);
				}

				DrawLightVolume(RHICmdList);
			});
	}

	/**
	 * Pass 2: a fullscreen triangle whose stencil test admits only tagged
	 * pixels, clearing the tag as it goes so the next light starts clean.
	 *
	 * Deliberately not scissored or depth bounds tested: anything that stopped
	 * a tagged pixel from being rasterised here would also stop its tag from
	 * being cleared.
	 */
	template <typename TPixelShader>
	void AddStencilTaggedLightPass(
		FRDGBuilder& GraphBuilder,
		const FGlobalShaderMap* ShaderMap,
		const TShaderMapRef<TPixelShader>& PixelShader,
		typename TPixelShader::FParameters* PassParameters,
		const FIntRect& ViewRect)
	{
		GraphBuilder.AddPass(
			RDG_EVENT_NAME("SWRTSpotLight.Shade"),
			PassParameters,
			ERDGPassFlags::Raster,
			[PassParameters, PixelShader, ShaderMap, ViewRect](FRDGAsyncTask, FRHICommandList& RHICmdList)
			{
				FGraphicsPipelineStateInitializer GraphicsPSOInit;
				FPixelShaderUtils::InitFullscreenPipelineState(RHICmdList, ShaderMap, PixelShader, GraphicsPSOInit);

				GraphicsPSOInit.BlendState = TStaticBlendState<CW_RGBA, BO_Add, BF_One, BF_One, BO_Add, BF_One, BF_One>::GetRHI();
				GraphicsPSOInit.RasterizerState = TStaticRasterizerState<FM_Solid, CM_None>::GetRHI();
				GraphicsPSOInit.DepthStencilState = TStaticDepthStencilState<
					false, CF_Always,
					true, CF_NotEqual, SO_Zero, SO_Zero, SO_Zero,
					true, CF_NotEqual, SO_Zero, SO_Zero, SO_Zero,
					STENCIL_SANDBOX_MASK, STENCIL_SANDBOX_MASK>::GetRHI();

				RHICmdList.SetViewport((float)ViewRect.Min.X, (float)ViewRect.Min.Y, 0.0f, (float)ViewRect.Max.X, (float)ViewRect.Max.Y, 1.0f);
				SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit, 0);
				SetShaderParameters(RHICmdList, PixelShader, PixelShader.GetPixelShader(), *PassParameters);
				FPixelShaderUtils::DrawFullscreenTriangle(RHICmdList);
			});
	}

	/*=========================================================================
		THE LIGHT PASS
	=========================================================================*/
	void FillLightParameters(
		FRDGBuilder& GraphBuilder,
		const FViewInfo& View,
		const FSWRTSpotLightRenderData& Light,
		float FadeFactor,
		FSWRTSpotLightPS::FParameters& Parameters)
	{
		const FVector PreViewTranslation = View.ViewMatrices.GetPreViewTranslation();

		Parameters.View = View.ViewUniformBuffer;

		// Scene colour is deliberately not requested: this pass writes it, and
		// RDG rejects a resource bound for read and write in one pass.
		Parameters.SceneTextures = CreateSceneTextureShaderParameters(
			GraphBuilder,
			View,
			ESceneTextureSetupMode::SceneDepth |
			ESceneTextureSetupMode::GBufferA |
			ESceneTextureSetupMode::GBufferB |
			ESceneTextureSetupMode::GBufferC |
			ESceneTextureSetupMode::GBufferD |
			ESceneTextureSetupMode::GBufferE |
			ESceneTextureSetupMode::GBufferF |
			ESceneTextureSetupMode::CustomDepth);

		// GetDeferredLightParameters. The subtraction happens in doubles before
		// narrowing so the light does not jitter far from the world origin.
		Parameters.SWRTLightTranslatedWorldPosition = FVector3f(Light.Position + PreViewTranslation);
		Parameters.SWRTLightInvRadius = 1.0f / FMath::Max(0.00001f, Light.Radius);

		// FViewInfo hides FSceneView's exported accessor with a non exported one
		// of the same name, so it is called through the base class.
		const float Exposure = static_cast<const FSceneView&>(View).GetLastEyeAdaptationExposure();
		const float LightExposureScale = FLightRenderParameters::GetLightExposureScale(Exposure, Light.InverseExposureBlend);
		const FLinearColor LightColor = Light.Color * LightExposureScale * FadeFactor;
		Parameters.SWRTLightColor = FVector3f(LightColor.R, LightColor.G, LightColor.B);

		Parameters.SWRTLightFalloffExponent = Light.FalloffExponent;
		Parameters.SWRTLightDirection = FVector3f(-Light.Direction);
		Parameters.SWRTLightTangent = FVector3f(Light.Tangent);
		Parameters.SWRTLightSpotAngles = FVector2f(Light.CosOuterCone, Light.InvCosConeDifference);
		Parameters.SWRTLightSourceRadius = Light.SourceRadius;
		Parameters.SWRTLightSoftSourceRadius = Light.SoftSourceRadius;
		Parameters.SWRTLightSourceLength = Light.SourceLength;
		Parameters.SWRTLightSpecularScale = Light.SpecularScale;
		Parameters.SWRTLightDiffuseScale = Light.DiffuseScale;
		Parameters.SWRTContactShadowLength = Light.ContactShadowLength;
		Parameters.SWRTContactShadowCastingIntensity = Light.ContactShadowCastingIntensity;
		Parameters.SWRTContactShadowNonCastingIntensity = Light.ContactShadowNonCastingIntensity;

		// GetShadowedBits: 3 when the light casts dynamic shadows, which also
		// enables contact shadows; 0 otherwise, in which case the engine applies
		// ambient occlusion to direct light instead of a shadow term.
		Parameters.SWRTShadowedBits =
			(Light.bCastShadows && View.Family->EngineShowFlags.DynamicShadows) ? 3u : 0u;
	}

	void AddSWRTSpotLightPass(
		FRDGBuilder& GraphBuilder,
		const FViewInfo& View,
		FRDGTextureRef SceneColorTexture,
		FRDGTextureRef SceneDepthTexture,
		const FSWRTSpotLightRenderData& Light,
		float FadeFactor,
		FRDGTextureRef ShadowMask,
		const FSWRTLightVolume& Volume)
	{
		const FRDGSystemTextures& SystemTextures = FRDGSystemTextures::Get(GraphBuilder);
		const bool bShadowed = ShadowMask != nullptr;

		FSWRTSpotLightPS::FParameters* PassParameters = GraphBuilder.AllocParameters<FSWRTSpotLightPS::FParameters>();
		FillLightParameters(GraphBuilder, View, Light, FadeFactor, *PassParameters);

		// White, not black: the dummy stands in for nothing occluded, so a
		// permutation that read it by mistake shows an unshadowed light rather
		// than a black one. Wrong either way, but one of them is visible.
		PassParameters->SWRTShadowMask = bShadowed ? ShadowMask : SystemTextures.White;
		PassParameters->SWRTDebugMode = (uint32)SWRTSpot::GetDebugMode();

		PassParameters->RenderTargets[0] = FRenderTargetBinding(SceneColorTexture, ERenderTargetLoadAction::ELoad);

		FSWRTSpotLightPS::FPermutationDomain PermutationVector;
		PermutationVector.Set<FSWRTSpotLightPS::FShadowed>(bShadowed);

		const FIntRect ViewRect = View.ViewRect;
		TShaderMapRef<FSWRTLightVS> VertexShader(View.ShaderMap);

		AddStencilTagPass(GraphBuilder, View, SceneDepthTexture, Volume, VertexShader, ViewRect);

		// Stencil writable so the shade can clear the tag; depth stays read only,
		// which is what allows scene depth to be sampled through SceneTextures in
		// the same pass.
		PassParameters->RenderTargets.DepthStencil = FDepthStencilBinding(
			SceneDepthTexture,
			ERenderTargetLoadAction::ELoad,
			ERenderTargetLoadAction::ELoad,
			FExclusiveDepthStencil::DepthRead_StencilWrite);

		TShaderMapRef<FSWRTSpotLightPS> PixelShader(View.ShaderMap, PermutationVector);
		AddStencilTaggedLightPass(GraphBuilder, View.ShaderMap, PixelShader, PassParameters, ViewRect);
	}
}

void SWRTSpot::RenderSWRTSpotLights(
	FRDGBuilder& GraphBuilder,
	FViewInfo& View,
	FRDGTextureRef SceneColorTexture,
	FRDGTextureRef SceneDepthTexture,
	TConstArrayView<FSWRTSpotLightRenderData> Lights)
{
	if (Lights.Num() == 0 || SceneColorTexture == nullptr || SceneDepthTexture == nullptr)
	{
		return;
	}

	// Substrate reads a different GBuffer layout and lights through its own
	// pass, so this plugin has nothing it can decode.
	if (Substrate::IsSubstrateEnabled())
	{
		static bool bWarned = false;
		if (!bWarned)
		{
			bWarned = true;
			UE_LOG(LogTemp, Warning,
				TEXT("PrototypeSWRTShadows: SWRT spot lights are not supported with Substrate enabled and will not render."));
		}
		return;
	}

	RDG_EVENT_SCOPE(GraphBuilder, "SWRTSpotLights");

	// Every light that will draw, and the casters they need BETWEEN them.
	//
	// WHY THE CASTERS ARE UNIONED
	// The triangle cache is one global thing, and each trace zeroes
	// WrittenTriangles and then refills only the sources IT gathered. Give two
	// lights different caster lists and each marks the other's sources
	// unwritten, so an object that casts for both has its slot and its
	// triangle offset reshuffled between the two traces of a single frame. One
	// of them then walks a tree whose leaves point at the other's arrangement,
	// and its shadow appears and disappears frame to frame. It only shows with
	// more than one light, and only on the casters they share.
	//
	// Tracing every light against the union costs each one placements it
	// cannot reach. The per tile placement list culls those against the hull of
	// the tile and the light, so they cost a box test rather than a walk.
	struct FVisibleLight
	{
		const FSWRTSpotLightRenderData* Light = nullptr;
		float FadeFactor = 1.0f;
		bool bTrace = false;
	};

	TArray<FVisibleLight> Visible;
	Visible.Reserve(Lights.Num());

	TArray<const FPrimitiveSceneProxy*> UnionCasters;
	TSet<const FPrimitiveSceneProxy*> SeenCasters;

	for (const FSWRTSpotLightRenderData& Light : Lights)
	{
		FVisibleLight Entry;
		Entry.Light = &Light;
		if (!IsSWRTSpotLightVisibleInView(View, Light, Entry.FadeFactor))
		{
			continue;
		}

		if (Light.bCastShadows && View.Family->EngineShowFlags.DynamicShadows)
		{
			TArray<const FPrimitiveSceneProxy*> Casters;
			SWRTSpot::GatherSWRTCasters(GraphBuilder, View, Light, Casters);

			if (Casters.Num() > 0)
			{
				Entry.bTrace = true;
				for (const FPrimitiveSceneProxy* Proxy : Casters)
				{
					bool bAlreadySeen = false;
					SeenCasters.Add(Proxy, &bAlreadySeen);
					if (!bAlreadySeen)
					{
						UnionCasters.Add(Proxy);
					}
				}
			}
		}

		Visible.Add(Entry);
	}

	for (const FVisibleLight& Entry : Visible)
	{
		const FSWRTSpotLightRenderData& Light = *Entry.Light;
		FRDGTextureRef ShadowMask = nullptr;

		if (Entry.bTrace)
		{
			SWRTTrace::FTraceLight TraceLight;
			// Translated world, because that is the space the trace shader
			// reconstructs its surface positions in.
			TraceLight.TranslatedPosition =
				Light.Position + View.ViewMatrices.GetPreViewTranslation();
			TraceLight.WorldPosition = Light.Position;
			TraceLight.Direction = Light.Direction;
			TraceLight.Radius = Light.Radius;
			TraceLight.CosOuterCone = Light.CosOuterCone;

			ShadowMask = SWRTTrace::TickSoftwareTrace(
				GraphBuilder, View, SceneDepthTexture, UnionCasters, TraceLight);
		}

		const FSWRTLightVolume Volume = SetupLightVolume(View, Light);

		AddSWRTSpotLightPass(
			GraphBuilder, View, SceneColorTexture, SceneDepthTexture,
			Light, Entry.FadeFactor, ShadowMask, Volume);
	}
}

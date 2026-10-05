// SPDX-License-Identifier: Apache-2.0
//
// Render thread data and entry points for the SWRT spot light.
//
// Frame flow for one light:
//
//   Game thread   GatherSWRTSpotLights   components to FSWRTSpotLightRenderData
//   Render thread RenderSWRTSpotLights   once per view
//                   IsSWRTSpotLightVisibleInView   engine light culling
//                   GatherSWRTCasters              cone and visible region cull
//                   TickSoftwareTrace              BVH build and one ray a pixel
//                   AddSWRTSpotLightPass           stencil volume, then shade

#pragma once

#include "CoreMinimal.h"
#include "RenderGraphDefinitions.h"
#include "UObject/ObjectKey.h"

class FViewInfo;
class FPrimitiveSceneProxy;
class UWorld;

/**
 * Everything the render thread needs about one light, copied off the component
 * on the game thread. No UObject pointers, so it stays valid while the game
 * thread modifies or destroys the component.
 *
 * Field meanings follow FLightSceneProxy and FLightRenderParameters.
 */
struct FSWRTSpotLightRenderData
{
	/** Stable identity of the component. */
	FObjectKey LightKey;

	FVector Position = FVector::ZeroVector;

	/** Light to world without scale. Local X is the light's forward axis. */
	FMatrix LightToWorld = FMatrix::Identity;

	/** Inverse of LightToWorld with the translation removed. */
	FMatrix WorldToLightRotation = FMatrix::Identity;

	FVector Direction = FVector::ForwardVector;
	FVector Tangent = FVector::UpVector;

	/** Colour times brightness, with intensity units and temperature applied. */
	FLinearColor Color = FLinearColor::Black;
	float InverseExposureBlend = 0.0f;

	float Radius = 1.0f;
	/** 0 means inverse squared falloff, which is the engine's own encoding. */
	float FalloffExponent = 0.0f;

	float SourceRadius = 0.0f;
	float SoftSourceRadius = 0.0f;
	float SourceLength = 0.0f;
	float SpecularScale = 1.0f;
	float DiffuseScale = 1.0f;

	/** Clamped the way FSpotLightSceneProxy clamps them. */
	float OuterConeAngle = 0.0f;
	float CosOuterCone = -2.0f;
	float SinOuterCone = 0.0f;
	float InvCosConeDifference = 1.0f;

	/** FLightSceneProxy::GetBoundingSphere: the cone's bounding sphere. */
	FSphere Bounds = FSphere(ForceInit);

	float MaxDrawDistance = 0.0f;
	float MaxDistanceFadeRange = 0.0f;

	/** Negative when the length is in world units, as the engine encodes it. */
	float ContactShadowLength = 0.0f;
	float ContactShadowCastingIntensity = 1.0f;
	float ContactShadowNonCastingIntensity = 0.0f;

	bool bCastShadows = false;

	/** r.AllowDepthBoundsTest, read on the game thread. */
	bool bAllowDepthBoundsTest = true;
};

namespace SWRTSpot
{
	/** Game thread. Appends every visible, enabled SWRT spot light in the world. */
	void GatherSWRTSpotLights(UWorld* OwningWorld, TArray<FSWRTSpotLightRenderData>& OutLights);

	/**
	 * Render thread. Culls, traces and shades every light for one view,
	 * additively into scene colour.
	 */
	void RenderSWRTSpotLights(
		FRDGBuilder& GraphBuilder,
		FViewInfo& View,
		FRDGTextureRef SceneColorTexture,
		FRDGTextureRef SceneDepthTexture,
		TConstArrayView<FSWRTSpotLightRenderData> Lights);

	/**
	 * Render thread. The shadow casters this light keeps, which is also the set
	 * the BVH is built from.
	 *
	 * Two bounds, intersected: the cone frustum, and the convex hull of the bulb
	 * joined to the camera frustum. A caster that shadows a visible point lies
	 * on the segment from the bulb to that point, so anything outside that hull
	 * shadows only what the player cannot see. Conservative on both bounds, so
	 * it can keep a caster a tighter test would drop but never drop one that
	 * matters.
	 */
	void GatherSWRTCasters(
		FRDGBuilder& GraphBuilder,
		const FViewInfo& View,
		const FSWRTSpotLightRenderData& Light,
		TArray<const FPrimitiveSceneProxy*>& OutCasters);

	/** r.SWRTSpotLight.Debug, 0 when it holds a value no view uses. */
	int32 GetDebugMode();

	/**
	 * Which measurement the trace writes into the mask instead of a shadow
	 * term. 0 is a real shadow. The numbering is the trace shader's own.
	 */
	uint32 GetTraceView();
}

namespace SWRTTrace
{
	/** Everything the trace needs about the light, in translated world space. */
	struct FTraceLight
	{
		FVector TranslatedPosition = FVector::ZeroVector;

		/** World space, for projecting the light bounds to a screen rect. */
		FVector WorldPosition = FVector::ZeroVector;
		FVector Direction = FVector::ForwardVector;
		float Radius = 0.0f;
		float CosOuterCone = -1.0f;

		/**
		 * Ray start offset, a + b times view distance, in world units. The ray
		 * begins at a position reconstructed from the depth buffer, which does
		 * not land exactly on the triangle that wrote it, so starting at zero
		 * makes the surface shadow itself. The slope term exists because that
		 * reconstruction error grows with distance from the camera.
		 */
		float RayStartBias = 0.5f;
		float RayStartSlope = 0.002f;
	};

	/**
	 * Render thread. Traces one hard shadow ray per lit pixel against a two
	 * level BVH built from Casters alone, and returns an R8 mask where 0 is
	 * shadowed and 1 is lit.
	 *
	 * Returns null when nothing could be traced this frame. Null is not
	 * "everything is lit": the caller must shade the light unshadowed only
	 * because there was no caster to shadow with.
	 */
	FRDGTextureRef TickSoftwareTrace(
		FRDGBuilder& GraphBuilder,
		const FViewInfo& View,
		FRDGTextureRef SceneDepthTexture,
		TConstArrayView<const FPrimitiveSceneProxy*> Casters,
		const FTraceLight& Light);
}

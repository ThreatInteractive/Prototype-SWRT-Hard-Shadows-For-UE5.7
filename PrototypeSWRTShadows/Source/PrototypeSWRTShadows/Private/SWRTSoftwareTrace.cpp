// SPDX-License-Identifier: Apache-2.0
//
// The software ray traced shadow path: a two level BVH built on the GPU from
// one light's shadow casters, and one hard shadow ray per lit pixel traversed
// in plain compute. No hardware ray tracing anywhere.
//
// WHAT RUNS IN A FRAME
//   gather        each caster's vertex and index buffers, by index range
//   extract       a compute pass writes triangles into one shared buffer
//   build         a BVH per source, on worker threads, a few sources a frame
//   refit         the node bounds, on the GPU, every frame
//   top level     a BVH over the placements, on the render thread
//   tile list     per 8x8 tile, which placements its rays can reach
//   trace         one ray a pixel into an R8 mask: 0 shadowed, 1 lit
//
// THE TWO LEVELS
// Triangles are stored in their source's LOCAL space, so two placements of the
// same mesh share one copy of the geometry and one tree. The traversal
// transforms the RAY into each instance's frame instead, which is three dot
// products against the rows in SWRTInstanceTransforms. That record has a stride
// of five float4: rows 0 to 2 are world to local, row 3 is the frame minimum
// and row 4 the frame scale for the quantised node bounds.
//
// WHY THE BUILD IS SPREAD OVER FRAMES
// A sorted build is O(triangles log triangles), which is far too much to do in
// one frame when a large scene first comes into view. The sort and the node
// layout are both worker tasks, with a budget in sources and in triangles per
// frame, and a source keeps using the tree it already has until its new one
// lands. Nothing stalls on a build.
//
// THE SHADOW TERM
// One ray, cast and answered in world space, for every pixel the light reaches.
// Nothing is filtered, compared against a map, or carried over from an earlier
// frame.

#include "SWRTSpotLightRendering.h"

#include "CachedGeometry.h"
#include "DynamicRHI.h"
#include "GlobalShader.h"
#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/ScopeExit.h"
#include "RenderTimer.h"
#include "MeshBatch.h"
#include "PrimitiveSceneInfo.h"
#include "PrimitiveSceneProxy.h"
#include "RenderGraphBuilder.h"
#include "RHIGPUReadback.h"
#include "RenderGraphUtils.h"
#include "RenderUtils.h"
#include "ShaderParameterStruct.h"
#include "Tasks/Task.h"
#include "SkeletalMeshSceneProxy.h"
#include "SkeletalRenderPublic.h"
#include "VertexFactory.h"
#include "ScenePrivate.h"
#include "SceneRendering.h"
#include "Algo/Sort.h"

namespace SWRTTrace
{
	/*=========================================================================
		THE SHAPE OF THE TREE

		Three numbers decide it, and the traversal is written around them: a
		node holds four children and fetches all four at once, a leaf holds two
		triangles, and a tile's placement list is sixteen words long.
	=========================================================================*/

	/** Triangles a leaf holds. */
	static constexpr int32 GLeafTriangles = 2;

	/**
	 * Children an internal node holds. The traversal fetches this many at once
	 * and skips the ones past the node's count, which turns four dependent
	 * loads into one.
	 */
	static constexpr uint32 GNodeArity = 4;

	/**
	 * Words a tile's placement list occupies, the header included. A tile
	 * needing more than fits falls back to the full top level walk, so the
	 * list is a shortcut and never a cull.
	 */
	static constexpr uint32 GTileListStride = 16;

	static TAutoConsoleVariable<int32> CVarTraceHeatmapRange(
		TEXT("r.SWRTSpotLight.Trace.HeatmapRange"),
		42,
		TEXT("Node visits that paint full red in the ray iteration heatmap, which is\n")
		TEXT("r.SWRTSpotLight.Debug 3. Lower makes the picture hotter.\n")
		TEXT("Counts both the top level and the BLAS walk, so a ray that spent its cost\n")
		TEXT("descending placements looks as hot as one that spent it inside geometry.\n")
		TEXT("This has to follow the tree: every structural win makes it too high again, and\n")
		TEXT("a heatmap nobody can read reads as nothing being hot."),
		ECVF_RenderThreadSafe);
	/**
	 * The cone's screen bound, snapped outward to the 8x8 tile grid the trace
	 * dispatches and lists by.
	 *
	 * False when any sample is behind the camera, where the projection folds
	 * through infinity. Giving up has to WIDEN the rect, never shrink it, or
	 * shadows disappear silently.
	 */
	static bool ComputeConeScreenRect(
		const FViewInfo& View, const SWRTTrace::FTraceLight& Light,
		FIntRect& OutRect)
	{
		const FVector Apex = Light.WorldPosition;
		const FVector Direction = Light.Direction.GetSafeNormal();
		if (Direction.IsNearlyZero() || Light.Radius <= 0.0f)
		{
			return false;
		}
		const double CosOuter = FMath::Clamp((double)Light.CosOuterCone, -1.0, 1.0);
		const double SinOuter = FMath::Sqrt(FMath::Max(1.0 - CosOuter * CosOuter, 0.0));
		const FVector BaseCentre = Apex + Direction * ((double)Light.Radius * CosOuter);
		const double BaseRadius = (double)Light.Radius * SinOuter;
		FVector RingU, RingV;
		Direction.FindBestAxisVectors(RingU, RingV);
		const FMatrix ViewProjection = View.ViewMatrices.GetViewProjectionMatrix();
		const FIntRect ViewRect = View.ViewRect;
		FVector2D RectMin(TNumericLimits<double>::Max(), TNumericLimits<double>::Max());
		FVector2D RectMax(-TNumericLimits<double>::Max(), -TNumericLimits<double>::Max());
		constexpr int32 RingSteps = 16;
		for (int32 Point = 0; Point <= RingSteps; ++Point)
		{
			FVector World = Apex;
			if (Point > 0)
			{
				const double Angle = (2.0 * UE_DOUBLE_PI * (Point - 1)) / RingSteps;
				World = BaseCentre
					+ (RingU * FMath::Cos(Angle) + RingV * FMath::Sin(Angle)) * BaseRadius;
			}
			const FVector4 Clip = ViewProjection.TransformFVector4(FVector4(World, 1.0));
			if (Clip.W <= (double)UE_KINDA_SMALL_NUMBER)
			{
				return false;
			}
			const double NormalisedX = Clip.X / Clip.W;
			const double NormalisedY = Clip.Y / Clip.W;
			const double PixelX =
				ViewRect.Min.X + (NormalisedX * 0.5 + 0.5) * (double)ViewRect.Width();
			const double PixelY =
				ViewRect.Min.Y + (0.5 - NormalisedY * 0.5) * (double)ViewRect.Height();
			RectMin.X = FMath::Min(RectMin.X, PixelX);
			RectMin.Y = FMath::Min(RectMin.Y, PixelY);
			RectMax.X = FMath::Max(RectMax.X, PixelX);
			RectMax.Y = FMath::Max(RectMax.Y, PixelY);
		}
		constexpr int32 TileSize = 8;
		FIntRect Rect(
			FMath::FloorToInt(RectMin.X), FMath::FloorToInt(RectMin.Y),
			FMath::CeilToInt(RectMax.X), FMath::CeilToInt(RectMax.Y));
		Rect.Min.X = ViewRect.Min.X
			+ ((Rect.Min.X - ViewRect.Min.X) / TileSize) * TileSize;
		Rect.Min.Y = ViewRect.Min.Y
			+ ((Rect.Min.Y - ViewRect.Min.Y) / TileSize) * TileSize;
		Rect.Max.X = Rect.Min.X
			+ FMath::DivideAndRoundUp(Rect.Max.X - Rect.Min.X, TileSize) * TileSize;
		Rect.Max.Y = Rect.Min.Y
			+ FMath::DivideAndRoundUp(Rect.Max.Y - Rect.Min.Y, TileSize) * TileSize;
		Rect.Clip(ViewRect);
		if (Rect.Width() <= 0 || Rect.Height() <= 0)
		{
			return false;
		}
		OutRect = Rect;
		return true;
	}


	/** Mirrors SWRT_STACK_SIZE and SWRT_INSTANCE_STACK_SIZE, for the stats line. */
	static int32 GStackCapacity(bool bInstanceStack)
	{
		return bInstanceStack ? 8 : 16;
	}

	/** Floats a stored triangle occupies: a vertex and both edges, in three float4. */
	static constexpr uint32 GTriangleStride = 12;

	/**
	 * Grid the quantisation frame snaps to, so the node bounds keep meaning the
	 * same thing while the scene bounds wobble by a few units and the cached
	 * trees stay valid.
	 */
	static constexpr double GFrameSnapUnits = 256.0;

	static TRefCountPtr<FRDGPooledBuffer> GTraceTriangles;

	static TUniquePtr<FRHIGPUBufferReadback> GStatsReadback;
	static bool GStatsReadbackInFlight = false;
	static uint32 GStatsReadbackFrame = 0;

	static TUniquePtr<FRHIGPUBufferReadback> GOrderReadback;
	static bool GOrderReadbackInFlight = false;
	static uint32 GOrderReadbackTriangles = 0;
	static uint32 GOrderReadbackFrame = 0;

	static TUniquePtr<FRHIGPUBufferReadback> GCasterFeedbackReadback;
	static bool GCasterFeedbackInFlight = false;
	static uint32 GCasterFeedbackCount = 0;
	static uint32 GCasterFeedbackFrame = 0;

	static TArray<int32> GCasterFeedbackSlots;

	/*=========================================================================
		GEOMETRY

		One SOURCE is one index range of one vertex buffer, which is the unit a
		BVH is built for. One PLACEMENT is a source plus a transform. Two
		placements of the same mesh are two placements of one source.
	=========================================================================*/

	struct FTriangleSource
	{
		FRHIShaderResourceView* PositionSRV = nullptr;
		FRHIShaderResourceView* IndexSRV = nullptr;
		FRHIBuffer* IndexBuffer = nullptr;
		uint32 FirstIndex = 0;
		uint32 NumPrimitives = 0;
		uint32 MinVertexIndex = 0;
		uint32 MaxVertexIndex = 0;
		bool b32BitIndices = false;
		FMatrix LocalToWorld = FMatrix::Identity;
		FBox WorldBounds = FBox(ForceInit);
		FBox LocalBounds = FBox(ForceInit);
		bool bSkinned = false;
		const void* Instance = nullptr;
	};

	/**
	 * A source's identity across frames. Everything the trace keeps for a
	 * source, its triangles and its tree, is keyed on this, so a source that
	 * comes back into view does not pay for a rebuild.
	 */

	struct FCachedSourceKey
	{
		const void* PositionSRV = nullptr;
		const void* IndexSRV = nullptr;
		const void* IndexBuffer = nullptr;
		uint32 FirstIndex = 0;
		uint32 NumPrimitives = 0;
		const void* Instance = nullptr;
		bool operator==(const FCachedSourceKey& Other) const
		{
			return PositionSRV == Other.PositionSRV
				&& IndexSRV == Other.IndexSRV
				&& IndexBuffer == Other.IndexBuffer
				&& FirstIndex == Other.FirstIndex
				&& NumPrimitives == Other.NumPrimitives
				&& Instance == Other.Instance;
		}
		friend uint32 GetTypeHash(const FCachedSourceKey& Key)
		{
			return HashCombine(
				HashCombine(::GetTypeHash(Key.PositionSRV), ::GetTypeHash(Key.IndexSRV)),
				HashCombine(::GetTypeHash(Key.IndexBuffer),
					HashCombine(::GetTypeHash(Key.FirstIndex),
						HashCombine(::GetTypeHash(Key.NumPrimitives), ::GetTypeHash(Key.Instance)))));
		}
	};

	struct FGatheredInstance
	{
		int32 SourceIndex = INDEX_NONE;
		FMatrix LocalToWorld = FMatrix::Identity;
		FBox WorldBounds = FBox(ForceInit);
		bool bSkinned = false;
	};

	struct FGatheredGeometry
	{
		TArray<FTriangleSource> Sources;
		TArray<FGatheredInstance> Instances;
		int32 NumStaticCasters = 0;
		int32 NumSkinnedCasters = 0;
		int32 NumSkippedCasters = 0;
		uint32 TotalTriangles = 0;
	};

	static FBox MakeBlasLocalBounds(const FPrimitiveSceneProxy* Proxy)
	{
		FBox Local = Proxy->GetLocalBounds().GetBox();
		if (!Local.IsValid || Local.GetSize().GetMin() <= 0.0)
		{
			Local = FBox(FVector(-1.0), FVector(1.0));
		}
		return Local;
	}

	static bool GatherStaticCaster(const FPrimitiveSceneProxy* Proxy, FGatheredGeometry& Out)
	{
		FPrimitiveSceneInfo* SceneInfo = Proxy->GetPrimitiveSceneInfo();
		if (SceneInfo == nullptr)
		{
			return false;
		}
		int8 MinLOD = 0;
		int8 MaxLOD = 0;
		SceneInfo->GetStaticMeshesLODRange(MinLOD, MaxLOD);
		const FMeshBatch* Batch = SceneInfo->GetMeshBatch(MinLOD);
		if (Batch == nullptr || Batch->VertexFactory == nullptr)
		{
			return false;
		}
		FRHIShaderResourceView* PositionSRV = Batch->VertexFactory->GetTriangleSortingPositionSRV();
		if (PositionSRV == nullptr)
		{
			return false;
		}
		bool bAny = false;
		for (const FMeshBatchElement& Element : Batch->Elements)
		{
			if (Element.NumPrimitives == 0 || Element.IndexBuffer == nullptr)
			{
				continue;
			}
			FTriangleSource& Source = Out.Sources.AddDefaulted_GetRef();
			Source.WorldBounds = Proxy->GetBounds().GetBox();
			Source.Instance = Proxy;
			Source.LocalBounds = MakeBlasLocalBounds(Proxy);
			Source.PositionSRV = PositionSRV;
			Source.IndexBuffer = Element.IndexBuffer->GetRHI();
			Source.b32BitIndices = Source.IndexBuffer != nullptr && Source.IndexBuffer->GetStride() == 4;
			Source.FirstIndex = Element.FirstIndex;
			Source.NumPrimitives = Element.NumPrimitives;
			Source.MinVertexIndex = Element.MinVertexIndex;
			Source.MaxVertexIndex = Element.MaxVertexIndex;
			Source.LocalToWorld = Proxy->GetLocalToWorld();
			Out.TotalTriangles += Element.NumPrimitives;
			bAny = true;
		}
		return bAny;
	}

	static bool GatherSkinnedCaster(
		FRDGBuilder& GraphBuilder, const FSkeletalMeshSceneProxy* Proxy, FGatheredGeometry& Out)
	{
		FCachedGeometry Cached;
		if (!Proxy->GetCachedGeometry(GraphBuilder, Cached))
		{
			return false;
		}
		bool bAny = false;
		for (const FCachedGeometry::Section& Section : Cached.Sections)
		{
			if (Section.NumPrimitives == 0)
			{
				continue;
			}
			FTriangleSource& Source = Out.Sources.AddDefaulted_GetRef();
			Source.WorldBounds = Proxy->GetBounds().GetBox();
			Source.Instance = Proxy;
			Source.LocalBounds = MakeBlasLocalBounds(Proxy);
			Source.PositionSRV = Section.PositionBuffer;
			Source.IndexSRV = Section.IndexBuffer;
			Source.FirstIndex = Section.IndexBaseIndex;
			Source.NumPrimitives = Section.NumPrimitives;
			Source.MinVertexIndex = Section.VertexBaseIndex;
			Source.MaxVertexIndex = Section.VertexBaseIndex + Section.NumVertices;
			Source.bSkinned = true;
			Source.LocalToWorld = Cached.LocalToWorld.ToMatrixWithScale();
			Out.TotalTriangles += Section.NumPrimitives;
			bAny = true;
		}
		return bAny;
	}

	void GatherCasterGeometry(
		FRDGBuilder& GraphBuilder,
		TConstArrayView<const FPrimitiveSceneProxy*> Casters,
		FGatheredGeometry& Out)
	{
		Out.Sources.Reserve(Casters.Num());
		for (const FPrimitiveSceneProxy* Proxy : Casters)
		{
			if (Proxy == nullptr)
			{
				continue;
			}
			const bool bSkinned = Proxy->IsSkinnedMesh() && !Proxy->IsNaniteMesh();
			bool bGathered = false;
			if (bSkinned)
			{
				bGathered = GatherSkinnedCaster(
					GraphBuilder, static_cast<const FSkeletalMeshSceneProxy*>(Proxy), Out);
				Out.NumSkinnedCasters += bGathered ? 1 : 0;
			}
			else
			{
				bGathered = GatherStaticCaster(Proxy, Out);
				Out.NumStaticCasters += bGathered ? 1 : 0;
			}
			Out.NumSkippedCasters += bGathered ? 0 : 1;
		}
	}

	class FSWRTTriangleExtractCS : public FGlobalShader
	{
		DECLARE_GLOBAL_SHADER(FSWRTTriangleExtractCS);
		SHADER_USE_PARAMETER_STRUCT(FSWRTTriangleExtractCS, FGlobalShader);
		BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
			SHADER_PARAMETER_SRV(Buffer<float>, SWRTExPositions)
			SHADER_PARAMETER_SRV(Buffer<uint>, SWRTExIndices)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<float4>, SWRTExOutTriangles)
			SHADER_PARAMETER(uint32, SWRTExFirstIndex)
			SHADER_PARAMETER(uint32, SWRTExNumPrimitives)
			SHADER_PARAMETER(uint32, SWRTExOutputTriangleOffset)
			SHADER_PARAMETER(FMatrix44f, SWRTExLocalToWorld)
			SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, SWRTExOrder)
			SHADER_PARAMETER(uint32, SWRTExOrderOffset)
		END_SHADER_PARAMETER_STRUCT()
		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
		{
			return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
		}
		static constexpr uint32 ThreadGroupSize = 64;
	};

	IMPLEMENT_GLOBAL_SHADER(FSWRTTriangleExtractCS, "/Plugins/PrototypeSWRTShadows/SWRTTriangleExtractCS.usf", "MainCS", SF_Compute);

	static FRDGBufferRef ExtractTriangles(
		FRDGBuilder& GraphBuilder,
		const FViewInfo& View,
		const FGatheredGeometry& Gathered,
		FRDGBufferRef TriangleBuffer,
		TConstArrayView<uint32> TriangleOffsets,
		TConstArrayView<bool> DirtySources,
		FRDGBufferRef OrderBuffer,
		TConstArrayView<uint32> SourceOrderOffsets,
		TConstArrayView<uint32> SourceEntryCounts,
		TArray<uint32>& OutWrittenTriangles)
	{
		OutWrittenTriangles.Init(0, Gathered.Sources.Num());
		if (Gathered.TotalTriangles == 0 || TriangleBuffer == nullptr)
		{
			return nullptr;
		}
		FRDGBufferUAVRef TriangleUAV = GraphBuilder.CreateUAV(
			TriangleBuffer, PF_A32B32G32R32F);
		FRDGBufferRef OrderToBind = OrderBuffer;
		if (OrderToBind == nullptr)
		{
			const uint32 Dummy = 0;
			OrderToBind = CreateStructuredBuffer(
				GraphBuilder, TEXT("SWRT.TriangleOrderDummy"),
				sizeof(uint32), 1, &Dummy, sizeof(Dummy));
		}
		FRDGBufferSRVRef OrderSRV = GraphBuilder.CreateSRV(OrderToBind, PF_R32_UINT);
		TShaderMapRef<FSWRTTriangleExtractCS> ComputeShader(
			GetGlobalShaderMap(View.GetFeatureLevel()));
		TArray<FShaderResourceViewRHIRef>& OwnedIndexViews =
			*GraphBuilder.AllocObject<TArray<FShaderResourceViewRHIRef>>();
		OwnedIndexViews.Reserve(Gathered.Sources.Num());

		// One scope over every source, for the same reason as the refit.
		RDG_EVENT_SCOPE(GraphBuilder, "SWRT.TriangleExtract");

		for (int32 SourceIndex = 0; SourceIndex < Gathered.Sources.Num(); ++SourceIndex)
		{
			const FTriangleSource& Source = Gathered.Sources[SourceIndex];
			const uint32 TriangleOffset = TriangleOffsets[SourceIndex];
			if (!DirtySources[SourceIndex])
			{
				OutWrittenTriangles[SourceIndex] =
					SourceEntryCounts.IsValidIndex(SourceIndex) && SourceEntryCounts[SourceIndex] > 0
						? SourceEntryCounts[SourceIndex]
						: Source.NumPrimitives;
				continue;
			}
			FRHIShaderResourceView* IndexSRV = Source.IndexSRV;
			if (IndexSRV == nullptr && Source.IndexBuffer != nullptr)
			{
				FShaderResourceViewRHIRef& Created = OwnedIndexViews.AddDefaulted_GetRef();
				Created = GraphBuilder.RHICmdList.CreateShaderResourceView(
					Source.IndexBuffer,
					FRHIViewDesc::CreateBufferSRV()
						.SetType(FRHIViewDesc::EBufferType::Typed)
						.SetFormat(Source.b32BitIndices ? PF_R32_UINT : PF_R16_UINT));
				IndexSRV = Created.GetReference();
			}
			if (Source.PositionSRV == nullptr || IndexSRV == nullptr)
			{
				continue;
			}
			FSWRTTriangleExtractCS::FParameters* Parameters =
				GraphBuilder.AllocParameters<FSWRTTriangleExtractCS::FParameters>();
			Parameters->SWRTExPositions = Source.PositionSRV;
			Parameters->SWRTExIndices = IndexSRV;
			Parameters->SWRTExOutTriangles = TriangleUAV;
			Parameters->SWRTExFirstIndex = Source.FirstIndex;
			Parameters->SWRTExNumPrimitives = SourceEntryCounts.IsValidIndex(SourceIndex) && SourceEntryCounts[SourceIndex] > 0 ? SourceEntryCounts[SourceIndex] : Source.NumPrimitives;
			Parameters->SWRTExOutputTriangleOffset = TriangleOffset;
			Parameters->SWRTExLocalToWorld = FMatrix44f(FMatrix::Identity);
			Parameters->SWRTExOrder = OrderSRV;
			Parameters->SWRTExOrderOffset = SourceOrderOffsets.IsValidIndex(SourceIndex)
				? SourceOrderOffsets[SourceIndex]
				: 0xFFFFFFFFu;
			FComputeShaderUtils::AddPass(
				GraphBuilder,
				RDG_EVENT_NAME("SWRT.TriangleExtract %u tris", Source.NumPrimitives),
				ComputeShader,
				Parameters,
				FIntVector(FMath::DivideAndRoundUp(Parameters->SWRTExNumPrimitives, FSWRTTriangleExtractCS::ThreadGroupSize), 1, 1));
			OutWrittenTriangles[SourceIndex] = Parameters->SWRTExNumPrimitives;
		}
		return TriangleBuffer;
	}

	/*=========================================================================
		NODE ENCODING

		A node is one uint4: three components of quantised bounds and one
		descriptor holding the first child or triangle, the count, and the leaf
		bit. 25 bits of index, 6 of count, 1 of leaf.
	=========================================================================*/
	static constexpr uint32 GNodeCountShift = 25;

	static constexpr uint32 GNodeCountMask = 0x3Fu;

	static constexpr uint32 GNodeLeafBit = 0x80000000u;

	static constexpr uint32 GNodeFirstMask = 0x01FFFFFFu;

	static constexpr int32 GMaxLeafSize = (int32)GNodeCountMask;

	static uint32 PackNodeDesc(uint32 First, uint32 Count, bool bLeaf)
	{
		checkSlow(First <= GNodeFirstMask && Count <= GNodeCountMask);
		return (First & GNodeFirstMask)
			| ((Count & GNodeCountMask) << GNodeCountShift)
			| (bLeaf ? GNodeLeafBit : 0u);
	}

	/*=========================================================================
		TOP LEVEL

		A BVH over the PLACEMENTS, built on the render thread because it changes
		every time anything moves. Small enough that a full sorted build costs
		less than a refit would.
	=========================================================================*/

	struct FInstanceTree
	{
		TArray<FUintVector4> Nodes;
		TArray<int32> Order;
		int32 Root = INDEX_NONE;
		int32 Depth = 0;
	};

	static uint32 MortonExpandBits(uint32 Value)
	{
		Value = (Value * 0x00010001u) & 0xFF0000FFu;
		Value = (Value * 0x00000101u) & 0x0F00F00Fu;
		Value = (Value * 0x00000011u) & 0xC30C30C3u;
		Value = (Value * 0x00000005u) & 0x49249249u;
		return Value;
	}

	static uint32 MortonCode(const FVector3f& Normalised)
	{
		const uint32 X = (uint32)FMath::Clamp(Normalised.X * 1023.0f, 0.0f, 1023.0f);
		const uint32 Y = (uint32)FMath::Clamp(Normalised.Y * 1023.0f, 0.0f, 1023.0f);
		const uint32 Z = (uint32)FMath::Clamp(Normalised.Z * 1023.0f, 0.0f, 1023.0f);
		return (MortonExpandBits(X) << 2) | (MortonExpandBits(Y) << 1) | MortonExpandBits(Z);
	}

	static FUintVector4 PackQuantisedNode(const FUintVector3& QMin, const FUintVector3& QMax, uint32 Desc)
	{
		return FUintVector4(
			QMin.X | (QMin.Y << 16),
			QMin.Z | (QMax.X << 16),
			QMax.Y | (QMax.Z << 16),
			Desc);
	}

	struct FInstanceSplitRange
	{
		int32 Start = 0;
		int32 Count = 0;
		int32 NodeIndex = INDEX_NONE;
		int32 Depth = 0;
	};

	static int32 MedianSplitInstances(
		TConstArrayView<FBox> InstanceBounds,
		TArray<int32>& Order,
		int32 Start,
		int32 Count,
		int32 Axis)
	{
		MakeArrayView(Order.GetData() + Start, Count).Sort(
			[&InstanceBounds, Axis](int32 A, int32 B)
			{
				return InstanceBounds[A].GetCenter()[Axis] < InstanceBounds[B].GetCenter()[Axis];
			});
		return Count / 2;
	}

	static int32 SplitInstancesSAH(
		TConstArrayView<FBox> InstanceBounds,
		TArray<int32>& Order,
		int32 Start,
		int32 Count,
		bool bAllAxes)
	{
		if (Count <= 1)
		{
			return INDEX_NONE;
		}
		FBox CentroidBounds(ForceInit);
		for (int32 Index = Start; Index < Start + Count; ++Index)
		{
			CentroidBounds += InstanceBounds[Order[Index]].GetCenter();
		}
		const FVector CentroidSize = CentroidBounds.GetSize();
		const int32 LongestAxis = (CentroidSize.X > CentroidSize.Y)
			? ((CentroidSize.X > CentroidSize.Z) ? 0 : 2)
			: ((CentroidSize.Y > CentroidSize.Z) ? 1 : 2);
		if (CentroidSize[LongestAxis] < UE_SMALL_NUMBER)
		{
			return Count / 2;
		}
		auto SurfaceArea = [](const FBox& Box) -> double
		{
			if (!Box.IsValid)
			{
				return 0.0;
			}
			const FVector Size = Box.GetSize();
			return 2.0 * (Size.X * Size.Y + Size.Y * Size.Z + Size.Z * Size.X);
		};
		constexpr int32 NumBins = 16;
		double BestCost = TNumericLimits<double>::Max();
		int32 BestAxis = INDEX_NONE;
		double BestSplitAt = 0.0;
		const int32 FirstAxis = bAllAxes ? 0 : LongestAxis;
		const int32 LastAxis = bAllAxes ? 2 : LongestAxis;
		for (int32 Axis = FirstAxis; Axis <= LastAxis; ++Axis)
		{
			if (CentroidSize[Axis] < UE_SMALL_NUMBER)
			{
				continue;
			}
			FBox BinBounds[NumBins];
			int32 BinCounts[NumBins] = {};
			for (int32 Bin = 0; Bin < NumBins; ++Bin)
			{
				BinBounds[Bin] = FBox(ForceInit);
			}
			const double BinScale = NumBins / CentroidSize[Axis];
			for (int32 Index = Start; Index < Start + Count; ++Index)
			{
				const FBox& Box = InstanceBounds[Order[Index]];
				const int32 Bin = FMath::Clamp(
					(int32)((Box.GetCenter()[Axis] - CentroidBounds.Min[Axis]) * BinScale), 0, NumBins - 1);
				BinBounds[Bin] += Box;
				++BinCounts[Bin];
			}
			FBox RightBounds[NumBins];
			int32 RightCounts[NumBins] = {};
			FBox Accumulated(ForceInit);
			int32 AccumulatedCount = 0;
			for (int32 Bin = NumBins - 1; Bin >= 1; --Bin)
			{
				Accumulated += BinBounds[Bin];
				AccumulatedCount += BinCounts[Bin];
				RightBounds[Bin] = Accumulated;
				RightCounts[Bin] = AccumulatedCount;
			}
			Accumulated = FBox(ForceInit);
			AccumulatedCount = 0;
			for (int32 Bin = 0; Bin < NumBins - 1; ++Bin)
			{
				Accumulated += BinBounds[Bin];
				AccumulatedCount += BinCounts[Bin];
				if (AccumulatedCount == 0 || RightCounts[Bin + 1] == 0)
				{
					continue;
				}
				const double Cost =
					SurfaceArea(Accumulated) * AccumulatedCount
					+ SurfaceArea(RightBounds[Bin + 1]) * RightCounts[Bin + 1];
				if (Cost < BestCost)
				{
					BestCost = Cost;
					BestAxis = Axis;
					BestSplitAt = CentroidBounds.Min[Axis] + (Bin + 1) / BinScale;
				}
			}
		}
		if (BestAxis == INDEX_NONE)
		{
			return MedianSplitInstances(InstanceBounds, Order, Start, Count, LongestAxis);
		}
		int32 Left = Start;
		int32 Right = Start + Count - 1;
		while (Left <= Right)
		{
			if (InstanceBounds[Order[Left]].GetCenter()[BestAxis] < BestSplitAt)
			{
				++Left;
			}
			else
			{
				Order.Swap(Left, Right);
				--Right;
			}
		}
		const int32 LeftCount = Left - Start;
		return (LeftCount == 0 || LeftCount == Count)
			? MedianSplitInstances(InstanceBounds, Order, Start, Count, LongestAxis)
			: LeftCount;
	}

	static void BuildInstanceTreeSAH(
		TConstArrayView<FBox> InstanceBounds,
		const FVector3f& QuantMin,
		const FVector3f& QuantScale,
		bool bAllAxes,
		FInstanceTree& Out)
	{
		const int32 NumInstances = InstanceBounds.Num();
		Out.Order.SetNumUninitialized(NumInstances);
		for (int32 Index = 0; Index < NumInstances; ++Index)
		{
			Out.Order[Index] = Index;
		}
		Out.Nodes.Reserve(NumInstances * 2);
		Out.Nodes.AddUninitialized(1);
		Out.Root = 0;
		auto QuantiseRange = [&](int32 Start, int32 Count, uint32 Desc) -> FUintVector4
		{
			FBox Bounds(ForceInit);
			for (int32 Index = Start; Index < Start + Count; ++Index)
			{
				Bounds += InstanceBounds[Out.Order[Index]];
			}
			const FVector3f ScaledMin = ((FVector3f)Bounds.Min - QuantMin) * QuantScale;
			const FVector3f ScaledMax = ((FVector3f)Bounds.Max - QuantMin) * QuantScale;
			const FUintVector3 QMin(
				(uint32)FMath::Clamp(FMath::FloorToFloat(ScaledMin.X), 0.0f, 65535.0f),
				(uint32)FMath::Clamp(FMath::FloorToFloat(ScaledMin.Y), 0.0f, 65535.0f),
				(uint32)FMath::Clamp(FMath::FloorToFloat(ScaledMin.Z), 0.0f, 65535.0f));
			const FUintVector3 QMax(
				(uint32)FMath::Clamp(FMath::CeilToFloat(ScaledMax.X), 0.0f, 65535.0f),
				(uint32)FMath::Clamp(FMath::CeilToFloat(ScaledMax.Y), 0.0f, 65535.0f),
				(uint32)FMath::Clamp(FMath::CeilToFloat(ScaledMax.Z), 0.0f, 65535.0f));
			return PackQuantisedNode(QMin, QMax, Desc);
		};
		TArray<FInstanceSplitRange> Queue;
		Queue.Reserve(NumInstances);
		Queue.Add({ 0, NumInstances, 0, 1 });
		for (int32 Head = 0; Head < Queue.Num(); ++Head)
		{
			const FInstanceSplitRange Task = Queue[Head];
			Out.Depth = FMath::Max(Out.Depth, Task.Depth);
			if (Task.Count <= 1)
			{
				Out.Nodes[Task.NodeIndex] =
					QuantiseRange(Task.Start, 1, PackNodeDesc((uint32)Task.Start, 1, true));
				continue;
			}
			const int32 Middle = SplitInstancesSAH(InstanceBounds, Out.Order, Task.Start, Task.Count, bAllAxes);
			const int32 LeftCount = (Middle == INDEX_NONE) ? Task.Count : Middle;
			const int32 RightCount = Task.Count - LeftCount;
			int32 ChildStarts[4];
			int32 ChildCounts[4];
			int32 NumChildren = 0;
			auto SplitHalf = [&](int32 HalfStart, int32 HalfCount)
			{
				if (HalfCount <= 0)
				{
					return;
				}
				const int32 HalfMiddle = SplitInstancesSAH(InstanceBounds, Out.Order, HalfStart, HalfCount, bAllAxes);
				if (HalfMiddle == INDEX_NONE || HalfMiddle <= 0 || HalfMiddle >= HalfCount)
				{
					ChildStarts[NumChildren] = HalfStart;
					ChildCounts[NumChildren] = HalfCount;
					++NumChildren;
					return;
				}
				ChildStarts[NumChildren] = HalfStart;
				ChildCounts[NumChildren] = HalfMiddle;
				++NumChildren;
				ChildStarts[NumChildren] = HalfStart + HalfMiddle;
				ChildCounts[NumChildren] = HalfCount - HalfMiddle;
				++NumChildren;
			};
			SplitHalf(Task.Start, LeftCount);
			SplitHalf(Task.Start + LeftCount, RightCount);
			const int32 FirstChild = Out.Nodes.Num();
			Out.Nodes.AddUninitialized(NumChildren);
			Out.Nodes[Task.NodeIndex] = QuantiseRange(
				Task.Start, Task.Count,
				PackNodeDesc((uint32)FirstChild, (uint32)NumChildren, false));
			for (int32 Child = 0; Child < NumChildren; ++Child)
			{
				Queue.Add({ ChildStarts[Child], ChildCounts[Child], FirstChild + Child, Task.Depth + 1 });
			}
		}
	}

	static void CollapseInstanceTree(FInstanceTree& Out, uint32 Budget)
	{
		if (Budget <= 1 || Out.Nodes.Num() <= 1)
		{
			return;
		}
		for (int32 Index = Out.Nodes.Num() - 1; Index >= 0; --Index)
		{
			const uint32 Desc = Out.Nodes[Index].W;
			if ((Desc & GNodeLeafBit) != 0)
			{
				continue;
			}
			const uint32 FirstChild = Desc & GNodeFirstMask;
			const uint32 ChildCount = (Desc >> GNodeCountShift) & GNodeCountMask;
			if (ChildCount < 2 || ChildCount > Budget
				|| (int32)(FirstChild + ChildCount) > Out.Nodes.Num())
			{
				continue;
			}
			uint32 FirstInstance = MAX_uint32;
			uint32 LastInstance = 0;
			bool bAllLeaves = true;
			for (uint32 Child = 0; Child < ChildCount; ++Child)
			{
				const uint32 ChildDesc = Out.Nodes[FirstChild + Child].W;
				if ((ChildDesc & GNodeLeafBit) == 0)
				{
					bAllLeaves = false;
					break;
				}
				const uint32 ChildFirst = ChildDesc & GNodeFirstMask;
				const uint32 ChildCountInstances = (ChildDesc >> GNodeCountShift) & GNodeCountMask;
				FirstInstance = FMath::Min(FirstInstance, ChildFirst);
				LastInstance = FMath::Max(LastInstance, ChildFirst + ChildCountInstances);
			}
			if (!bAllLeaves)
			{
				continue;
			}
			const uint32 Span = LastInstance - FirstInstance;
			if (Span > Budget || Span > (uint32)GMaxLeafSize || Span < 2)
			{
				continue;
			}
			Out.Nodes[Index].W = PackNodeDesc(FirstInstance, Span, true);
		}
	}

	static void BuildInstanceTree(
		TConstArrayView<FBox> InstanceBounds,
		uint32 Arity,
		const FVector3f& QuantMin,
		const FVector3f& QuantScale,
		const FVector& LightPosition,
		FInstanceTree& Out)
	{
		const int32 NumInstances = InstanceBounds.Num();
		if (NumInstances == 0)
		{
			return;
		}
		const int32 SAHMode = 2;
		if (SAHMode != 0)
		{
			BuildInstanceTreeSAH(
				InstanceBounds, QuantMin, QuantScale, SAHMode >= 2, Out);
			CollapseInstanceTree(
				Out,
				(uint32)FMath::Clamp(
					0, 1, GMaxLeafSize));
			return;
		}
		FBox Total(ForceInit);
		for (const FBox& Box : InstanceBounds)
		{
			Total += Box;
		}
		if (!Total.IsValid)
		{
			Total = FBox(FVector(-1.0), FVector(1.0));
		}
		const FVector TotalSize = Total.GetSize();
		TArray<uint32> Codes;
		Codes.SetNumUninitialized(NumInstances);
		Out.Order.Reserve(NumInstances);
		for (int32 Index = 0; Index < NumInstances; ++Index)
		{
			const FVector Centre = InstanceBounds[Index].GetCenter() - Total.Min;
			Codes[Index] = MortonCode(FVector3f(
				(float)(Centre.X / FMath::Max(TotalSize.X, 1.0)),
				(float)(Centre.Y / FMath::Max(TotalSize.Y, 1.0)),
				(float)(Centre.Z / FMath::Max(TotalSize.Z, 1.0))));
			Out.Order.Add(Index);
		}
		Out.Order.Sort([&Codes](int32 A, int32 B) { return Codes[A] < Codes[B]; });
		TArray<FUintVector3> Mins;
		TArray<FUintVector3> Maxs;
		Mins.Reserve(NumInstances * 2);
		Maxs.Reserve(NumInstances * 2);
		TArray<int32> PreviousLevel;
		PreviousLevel.Reserve(NumInstances);
		for (int32 Slot = 0; Slot < NumInstances; ++Slot)
		{
			const FBox& Box = InstanceBounds[Out.Order[Slot]];
			const FVector3f ScaledMin = ((FVector3f)Box.Min - QuantMin) * QuantScale;
			const FVector3f ScaledMax = ((FVector3f)Box.Max - QuantMin) * QuantScale;
			const FUintVector3 QMin(
				(uint32)FMath::Clamp(FMath::FloorToFloat(ScaledMin.X), 0.0f, 65535.0f),
				(uint32)FMath::Clamp(FMath::FloorToFloat(ScaledMin.Y), 0.0f, 65535.0f),
				(uint32)FMath::Clamp(FMath::FloorToFloat(ScaledMin.Z), 0.0f, 65535.0f));
			const FUintVector3 QMax(
				(uint32)FMath::Clamp(FMath::CeilToFloat(ScaledMax.X), 0.0f, 65535.0f),
				(uint32)FMath::Clamp(FMath::CeilToFloat(ScaledMax.Y), 0.0f, 65535.0f),
				(uint32)FMath::Clamp(FMath::CeilToFloat(ScaledMax.Z), 0.0f, 65535.0f));
			PreviousLevel.Add(Out.Nodes.Num());
			Out.Depth = 1;
			Mins.Add(QMin);
			Maxs.Add(QMax);
			Out.Nodes.Add(PackQuantisedNode(QMin, QMax, PackNodeDesc((uint32)Slot, 1, true)));
		}
		while (PreviousLevel.Num() > 1)
		{
			TArray<int32> ThisLevel;
			ThisLevel.Reserve(FMath::DivideAndRoundUp((uint32)PreviousLevel.Num(), Arity));
			for (int32 First = 0; First < PreviousLevel.Num(); First += (int32)Arity)
			{
				const int32 Count = FMath::Min((int32)Arity, PreviousLevel.Num() - First);
				FUintVector3 MergedMin(MAX_uint32, MAX_uint32, MAX_uint32);
				FUintVector3 MergedMax(0, 0, 0);
				for (int32 Child = 0; Child < Count; ++Child)
				{
					const int32 ChildNode = PreviousLevel[First + Child];
					MergedMin.X = FMath::Min(MergedMin.X, Mins[ChildNode].X);
					MergedMin.Y = FMath::Min(MergedMin.Y, Mins[ChildNode].Y);
					MergedMin.Z = FMath::Min(MergedMin.Z, Mins[ChildNode].Z);
					MergedMax.X = FMath::Max(MergedMax.X, Maxs[ChildNode].X);
					MergedMax.Y = FMath::Max(MergedMax.Y, Maxs[ChildNode].Y);
					MergedMax.Z = FMath::Max(MergedMax.Z, Maxs[ChildNode].Z);
				}
				ThisLevel.Add(Out.Nodes.Num());
				Mins.Add(MergedMin);
				Maxs.Add(MergedMax);
				Out.Nodes.Add(PackQuantisedNode(
					MergedMin, MergedMax,
					PackNodeDesc((uint32)PreviousLevel[First], (uint32)Count, false)));
			}
			PreviousLevel = MoveTemp(ThisLevel);
			++Out.Depth;
		}
		Out.Root = PreviousLevel[0];
	}

	/*=========================================================================
		PER SOURCE BVH

		Built from the triangles themselves, sorted, on a worker thread. The
		builder is free to split a triangle's bounds between leaves, which is
		what keeps a long thin triangle from making every box around it useless.
	=========================================================================*/

	struct FNodeLayout
	{
		TArray<uint32> Descs;
		TArray<int32> Roots;
		TArray<uint32> LevelNodeIndices;
		TArray<FIntPoint> Levels;
		TArray<TArray<FIntPoint>> SourceLevelSlices;
		TArray<uint32> NodeSources;
		int32 TlasRoot = INDEX_NONE;
		int32 NumBlasLevels = 0;
		TArray<int32> InstanceSlots;
	};

	struct FSahBuildNode
	{
		uint32 FirstChild = 0;
		uint32 ChildCount = 0;
		uint32 FirstTriangle = 0;
		uint32 TriangleCount = 0;
		bool bLeaf = false;
		int32 Height = 0;
	};

	struct FTriangleShapeProbe
	{
		double SumRatio = 0.0;
		double MaxRatio = 0.0;
		int64 Count = 0;
		void Accumulate(const FTriangleShapeProbe& Other)
		{
			SumRatio += Other.SumRatio;
			MaxRatio = FMath::Max(MaxRatio, Other.MaxRatio);
			Count += Other.Count;
		}
		double Mean() const { return Count > 0 ? SumRatio / (double)Count : 0.0; }
	};

	struct FTriangleCache
	{
		TArray<FCachedSourceKey> Keys;
		TMap<FCachedSourceKey, int32> KeyToSlot;
		TArray<FMatrix> Transforms;
		TArray<uint32> TriangleOffsets;
		TArray<uint32> WrittenTriangles;
		TArray<uint32> SlotReserve;
		TArray<FBox> LocalBounds;
		TArray<FBox> ExactLocalBounds;
		uint32 TotalTriangles = 0;
		TArray<TArray<uint32>> TriangleOrders;
		TArray<TArray<FSahBuildNode>> TriangleTrees;
		TArray<uint32> OrderTriangleCount;
		int32 LayoutFreeTreeCount = 0;
		uint32 LayoutCollapseBudget = 1;
		TArray<uint8> OrderApplied;
		TArray<uint32> LastEnteredFrame;
		FTriangleShapeProbe ShapeProbe;
		uint32 CachedTriangleCount = 0;
		FBox AccumulatedBounds = FBox(ForceInit);
		FVector SnappedMin = FVector::ZeroVector;
		FVector SnappedSize = FVector::ZeroVector;
		FNodeLayout Layout;
		uint32 LeafSize = 0;
		uint32 Arity = 0;
		TArray<TArray<FSahBuildNode>> LayoutTrees;
		TArray<int32> LayoutRootLocal;
		TArray<uint64> LayoutTreeKey;
		TArray<uint32> TreeSerial;
		TRefCountPtr<FRDGPooledBuffer> Triangles;
		TRefCountPtr<FRDGPooledBuffer> Nodes;
		TRefCountPtr<FRDGPooledBuffer> Descs;
		TRefCountPtr<FRDGPooledBuffer> NodeSources;
	};

	static FTriangleCache GTriangleCache;

	static void BuildTopLevel(uint32 Arity, FNodeLayout& Out);

	struct FSahTriangle
	{
		FVector3f Centroid = FVector3f::ZeroVector;
		FVector3f Min = FVector3f::ZeroVector;
		FVector3f Max = FVector3f::ZeroVector;
		uint32 Triangle = 0;
		float Area = 0.0f;
	};

	static double HalfSurfaceArea(const FVector3f& Min, const FVector3f& Max)
	{
		const FVector3f Extent = Max - Min;
		if (Extent.X < 0.0f || Extent.Y < 0.0f || Extent.Z < 0.0f)
		{
			return 0.0;
		}
		return (double)Extent.X * (double)Extent.Y
			+ (double)Extent.Y * (double)Extent.Z
			+ (double)Extent.Z * (double)Extent.X;
	}

	static void BoundsOfRange(
		const TArray<FSahTriangle>& Triangles, const TArray<uint32>& Order,
		int32 Start, int32 Count, FVector3f& OutMin, FVector3f& OutMax)
	{
		OutMin = FVector3f(FLT_MAX);
		OutMax = FVector3f(-FLT_MAX);
		for (int32 Index = Start; Index < Start + Count; ++Index)
		{
			const FSahTriangle& Triangle = Triangles[Order[Index]];
			OutMin = FVector3f::Min(OutMin, Triangle.Min);
			OutMax = FVector3f::Max(OutMax, Triangle.Max);
		}
	}

	static void SortRangeSAH(
		const TArray<FSahTriangle>& Triangles,
		TArray<uint32>& Order,
		int32 Start,
		int32 Count,
		uint32 LeafSize,
		TArray<uint32>& Scratch,
		TArray<uint32>& Best)
	{
		if ((uint32)Count <= LeafSize || Count < 2)
		{
			return;
		}
		const int32 Mid = Count / 2;
		double BestCost = MAX_dbl;
		bool bHaveBest = false;
		for (int32 Axis = 0; Axis < 3; ++Axis)
		{
			Scratch.SetNumUninitialized(Count, EAllowShrinking::No);
			FMemory::Memcpy(Scratch.GetData(), Order.GetData() + Start, Count * sizeof(uint32));
			const FSahTriangle* TriangleData = Triangles.GetData();
			Algo::Sort(Scratch, [TriangleData, Axis](uint32 A, uint32 B)
			{
				return TriangleData[A].Centroid[Axis] < TriangleData[B].Centroid[Axis];
			});
			FVector3f LeftMin(FLT_MAX), LeftMax(-FLT_MAX), RightMin(FLT_MAX), RightMax(-FLT_MAX);
			for (int32 Index = 0; Index < Count; ++Index)
			{
				const FSahTriangle& Triangle = Triangles[Scratch[Index]];
				if (Index < Mid)
				{
					LeftMin = FVector3f::Min(LeftMin, Triangle.Min);
					LeftMax = FVector3f::Max(LeftMax, Triangle.Max);
				}
				else
				{
					RightMin = FVector3f::Min(RightMin, Triangle.Min);
					RightMax = FVector3f::Max(RightMax, Triangle.Max);
				}
			}
			const double Cost =
				HalfSurfaceArea(LeftMin, LeftMax) * (double)Mid
				+ HalfSurfaceArea(RightMin, RightMax) * (double)(Count - Mid);
			if (!bHaveBest || Cost < BestCost)
			{
				BestCost = Cost;
				bHaveBest = true;
				Best.SetNumUninitialized(Count, EAllowShrinking::No);
				FMemory::Memcpy(Best.GetData(), Scratch.GetData(), Count * sizeof(uint32));
			}
		}
		if (!bHaveBest)
		{
			return;
		}
		FMemory::Memcpy(Order.GetData() + Start, Best.GetData(), Count * sizeof(uint32));
		SortRangeSAH(Triangles, Order, Start, Mid, LeafSize, Scratch, Best);
		SortRangeSAH(Triangles, Order, Start + Mid, Count - Mid, LeafSize, Scratch, Best);
	}

	static constexpr uint32 GSplitDepthShift = 27;

	static constexpr uint32 GSplitIndexShift = 29;

	static constexpr uint32 GSplitTriangleMask = (1u << GSplitDepthShift) - 1u;

	static constexpr uint32 GMaxSplitDepth = 3;

	static uint32 PackSplitRef(uint32 Triangle, uint32 Depth, uint32 Index)
	{
		return (Triangle & GSplitTriangleMask)
			| ((Depth & 0x3u) << GSplitDepthShift)
			| ((Index & 0x7u) << GSplitIndexShift);
	}

	static void SubTriangle(
		FVector3f& A, FVector3f& B, FVector3f& C, uint32 Depth, uint32 Index)
	{
		for (uint32 Step = 0; Step < Depth; ++Step)
		{
			const float LenAB = (B - A).SizeSquared();
			const float LenBC = (C - B).SizeSquared();
			const float LenCA = (A - C).SizeSquared();
			const bool bTakeLeft = ((Index >> (Depth - 1u - Step)) & 1u) == 0u;
			if (LenAB >= LenBC && LenAB >= LenCA)
			{
				const FVector3f Mid = (A + B) * 0.5f;
				if (bTakeLeft) { B = Mid; } else { A = Mid; }
			}
			else if (LenBC >= LenCA)
			{
				const FVector3f Mid = (B + C) * 0.5f;
				if (bTakeLeft) { C = Mid; } else { B = Mid; }
			}
			else
			{
				const FVector3f Mid = (C + A) * 0.5f;
				if (bTakeLeft) { A = Mid; } else { C = Mid; }
			}
		}
	}

	static void LoadRefCorners(
		const float* SourceTriangles,
		uint32 StrideFloats,
		uint32 PackedRef,
		FVector3f& OutA, FVector3f& OutB, FVector3f& OutC)
	{
		const uint32 Triangle = PackedRef & GSplitTriangleMask;
		const float* T = SourceTriangles + (SIZE_T)Triangle * StrideFloats;
		OutA = FVector3f(T[0], T[1], T[2]);
		OutB = FVector3f(T[0] + T[4], T[1] + T[5], T[2] + T[6]);
		OutC = FVector3f(T[0] + T[8], T[1] + T[9], T[2] + T[10]);
		SubTriangle(
			OutA, OutB, OutC,
			(PackedRef >> GSplitDepthShift) & 0x3u,
			(PackedRef >> GSplitIndexShift) & 0x7u);
	}

	static double RefEmptiness(const FSahTriangle& Ref)
	{
		return Ref.Area > UE_SMALL_NUMBER
			? HalfSurfaceArea(Ref.Min, Ref.Max) / (double)Ref.Area
			: 0.0;
	}

	static void RebuildRefBounds(
		FSahTriangle& Ref,
		const float* SourceTriangles,
		uint32 StrideFloats)
	{
		FVector3f A, B, C;
		LoadRefCorners(SourceTriangles, StrideFloats, Ref.Triangle, A, B, C);
		Ref.Min = FVector3f::Min(A, FVector3f::Min(B, C));
		Ref.Max = FVector3f::Max(A, FVector3f::Max(B, C));
		Ref.Centroid = (A + B + C) / 3.0f;
		Ref.Area = 0.5f * FVector3f::CrossProduct(B - A, C - A).Size();
	}

	static void BuildFreeSahTree(
		TArray<FSahTriangle>& Triangles,
		TArray<uint32>& Order,
		uint32 LeafSize,
		uint32 Arity,
		TArray<FSahBuildNode>& OutTree);

	static void BuildTriangleOrderSAH(
		const float* Triangles, uint32 TriangleCount, uint32 LeafSize,
		TArray<uint32>& OutOrder, FBox& OutExactBounds, FTriangleShapeProbe& OutProbe,
		TArray<FSahBuildNode>* OutFreeTree, uint32 FreeArity,
		uint32 MaxReferences)
	{
		const float* SourceTriangles = Triangles;
		const uint32 TriangleStrideFloats = GTriangleStride;
		OutOrder.Reset();
		OutExactBounds = FBox(ForceInit);
		if (Triangles == nullptr || TriangleCount == 0)
		{
			return;
		}
		TArray<FSahTriangle> Reduced;
		Reduced.SetNumUninitialized(TriangleCount);
		const SIZE_T Stride = GTriangleStride;
		for (uint32 Index = 0; Index < TriangleCount; ++Index)
		{
			const float* T = Triangles + (SIZE_T)Index * Stride;
			const FVector3f A(T[0], T[1], T[2]);
			const FVector3f B(T[0] + T[4], T[1] + T[5], T[2] + T[6]);
			const FVector3f C(T[0] + T[8], T[1] + T[9], T[2] + T[10]);
			FSahTriangle& Out = Reduced[Index];
			Out.Min = FVector3f::Min(A, FVector3f::Min(B, C));
			Out.Max = FVector3f::Max(A, FVector3f::Max(B, C));
			Out.Centroid = (A + B + C) / 3.0f;
			Out.Triangle = Index;
			Out.Area = 0.5f * FVector3f::CrossProduct(B - A, C - A).Size();
			if (Out.Min.ContainsNaN() || Out.Max.ContainsNaN())
			{
				OutExactBounds = FBox(ForceInit);
				return;
			}
			OutExactBounds += FVector(Out.Min);
			OutExactBounds += FVector(Out.Max);
			const double TriangleArea = 0.5 * (double)FVector3f::CrossProduct(B - A, C - A).Size();
			if (TriangleArea > UE_DOUBLE_SMALL_NUMBER)
			{
				const double Ratio = HalfSurfaceArea(Out.Min, Out.Max) / TriangleArea;
				OutProbe.SumRatio += Ratio;
				OutProbe.MaxRatio = FMath::Max(OutProbe.MaxRatio, Ratio);
				++OutProbe.Count;
			}
		}
		OutOrder.SetNumUninitialized(TriangleCount);
		for (uint32 Index = 0; Index < TriangleCount; ++Index)
		{
			OutOrder[Index] = Index;
		}
		if (OutFreeTree != nullptr && MaxReferences > TriangleCount)
		{
			struct FSplitCandidate
			{
				int32 Reference = 0;
				double Emptiness = 0.0;
				bool operator<(const FSplitCandidate& Other) const
				{
					return Emptiness > Other.Emptiness;
				}
			};
			TArray<FSplitCandidate> Heap;
			Heap.Reserve(Reduced.Num());
			TArray<uint32> RefDepth;
			RefDepth.SetNumZeroed(Reduced.Num());
			for (int32 Index = 0; Index < Reduced.Num(); ++Index)
			{
				const double Emptiness = RefEmptiness(Reduced[Index]);
				if (Emptiness > 2.5)
				{
					Heap.Add({ Index, Emptiness });
				}
			}
			Heap.Heapify();
			while (Reduced.Num() < (int32)MaxReferences && Heap.Num() > 0)
			{
				FSplitCandidate Worst;
				Heap.HeapPop(Worst, EAllowShrinking::No);
				const uint32 Depth = RefDepth[Worst.Reference];
				if (Depth >= GMaxSplitDepth)
				{
					continue;
				}
				const uint32 Triangle = Reduced[Worst.Reference].Triangle & GSplitTriangleMask;
				const uint32 Index =
					(Reduced[Worst.Reference].Triangle >> GSplitIndexShift) & 0x7u;
				FSahTriangle Left = Reduced[Worst.Reference];
				FSahTriangle Right = Reduced[Worst.Reference];
				Left.Triangle = PackSplitRef(Triangle, Depth + 1, Index * 2u);
				Right.Triangle = PackSplitRef(Triangle, Depth + 1, Index * 2u + 1u);
				RebuildRefBounds(Left, SourceTriangles, TriangleStrideFloats);
				RebuildRefBounds(Right, SourceTriangles, TriangleStrideFloats);
				const double Before = HalfSurfaceArea(
					Reduced[Worst.Reference].Min, Reduced[Worst.Reference].Max);
				const double After =
					HalfSurfaceArea(Left.Min, Left.Max) + HalfSurfaceArea(Right.Min, Right.Max);
				if (After >= Before * 0.95)
				{
					continue;
				}
				const int32 RightIndex = Reduced.Num();
				Reduced[Worst.Reference] = Left;
				RefDepth[Worst.Reference] = Depth + 1;
				Reduced.Add(Right);
				RefDepth.Add(Depth + 1);
				const double LeftEmptiness = RefEmptiness(Left);
				const double RightEmptiness = RefEmptiness(Right);
				if (LeftEmptiness > 2.5)
				{
					Heap.HeapPush({ Worst.Reference, LeftEmptiness });
				}
				if (RightEmptiness > 2.5)
				{
					Heap.HeapPush({ RightIndex, RightEmptiness });
				}
			}
			OutOrder.SetNumUninitialized(Reduced.Num());
			for (int32 Index = 0; Index < Reduced.Num(); ++Index)
			{
				OutOrder[Index] = (uint32)Index;
			}
		}
		if (OutFreeTree != nullptr)
		{
			TArray<uint32> RefOrder = MoveTemp(OutOrder);
			BuildFreeSahTree(
				Reduced, RefOrder, FMath::Max(LeafSize, 1u), FMath::Clamp(FreeArity, 2u, 8u),
				*OutFreeTree);
			OutOrder.SetNumUninitialized(RefOrder.Num());
			for (int32 Position = 0; Position < RefOrder.Num(); ++Position)
			{
				OutOrder[Position] = Reduced[RefOrder[Position]].Triangle;
			}
			return;
		}
		TArray<uint32> Scratch;
		TArray<uint32> Best;
		Scratch.Reserve(TriangleCount);
		Best.Reserve(TriangleCount);
		SortRangeSAH(
			Reduced, OutOrder, 0, (int32)TriangleCount, FMath::Max(LeafSize, 1u), Scratch, Best);
	}

	static constexpr int32 GSahNumBins = 16;

	static int32 SplitRangeFreeSAH(
		TArray<FSahTriangle>& Triangles,
		TArray<uint32>& Order,
		int32 Start,
		int32 Count)
	{
		if (Count < 2)
		{
			return 0;
		}
		FVector3f CentroidMin(FLT_MAX);
		FVector3f CentroidMax(-FLT_MAX);
		for (int32 Index = Start; Index < Start + Count; ++Index)
		{
			const FVector3f& C = Triangles[Order[Index]].Centroid;
			CentroidMin = FVector3f::Min(CentroidMin, C);
			CentroidMax = FVector3f::Max(CentroidMax, C);
		}
		const FVector3f CentroidExtent = CentroidMax - CentroidMin;
		int32 BestAxis = INDEX_NONE;
		int32 BestBin = INDEX_NONE;
		double BestCost = MAX_dbl;
		for (int32 Axis = 0; Axis < 3; ++Axis)
		{
			const float Extent = CentroidExtent[Axis];
			if (Extent <= UE_SMALL_NUMBER)
			{
				continue;
			}
			FVector3f BinMin[GSahNumBins];
			FVector3f BinMax[GSahNumBins];
			int32 BinCount[GSahNumBins];
			for (int32 Bin = 0; Bin < GSahNumBins; ++Bin)
			{
				BinMin[Bin] = FVector3f(FLT_MAX);
				BinMax[Bin] = FVector3f(-FLT_MAX);
				BinCount[Bin] = 0;
			}
			const float Scale = (float)GSahNumBins / Extent;
			for (int32 Index = Start; Index < Start + Count; ++Index)
			{
				const FSahTriangle& Triangle = Triangles[Order[Index]];
				const int32 Bin = FMath::Clamp(
					(int32)((Triangle.Centroid[Axis] - CentroidMin[Axis]) * Scale),
					0, GSahNumBins - 1);
				BinMin[Bin] = FVector3f::Min(BinMin[Bin], Triangle.Min);
				BinMax[Bin] = FVector3f::Max(BinMax[Bin], Triangle.Max);
				++BinCount[Bin];
			}
			double RightArea[GSahNumBins];
			int32 RightCount[GSahNumBins];
			{
				FVector3f RunMin(FLT_MAX);
				FVector3f RunMax(-FLT_MAX);
				int32 Running = 0;
				for (int32 Bin = GSahNumBins - 1; Bin >= 0; --Bin)
				{
					if (BinCount[Bin] > 0)
					{
						RunMin = FVector3f::Min(RunMin, BinMin[Bin]);
						RunMax = FVector3f::Max(RunMax, BinMax[Bin]);
						Running += BinCount[Bin];
					}
					RightArea[Bin] = Running > 0 ? HalfSurfaceArea(RunMin, RunMax) : 0.0;
					RightCount[Bin] = Running;
				}
			}
			{
				FVector3f RunMin(FLT_MAX);
				FVector3f RunMax(-FLT_MAX);
				int32 Running = 0;
				for (int32 Bin = 0; Bin < GSahNumBins - 1; ++Bin)
				{
					if (BinCount[Bin] > 0)
					{
						RunMin = FVector3f::Min(RunMin, BinMin[Bin]);
						RunMax = FVector3f::Max(RunMax, BinMax[Bin]);
						Running += BinCount[Bin];
					}
					if (Running == 0 || RightCount[Bin + 1] == 0)
					{
						continue;
					}
					const double Cost =
						HalfSurfaceArea(RunMin, RunMax) * (double)Running
						+ RightArea[Bin + 1] * (double)RightCount[Bin + 1];
					if (Cost < BestCost)
					{
						BestCost = Cost;
						BestAxis = Axis;
						BestBin = Bin;
					}
				}
			}
		}
		if (BestAxis == INDEX_NONE)
		{
			return Count / 2;
		}
		const float Extent = CentroidExtent[BestAxis];
		const float Scale = (float)GSahNumBins / Extent;
		const float AxisMin = CentroidMin[BestAxis];
		int32 Left = Start;
		int32 Right = Start + Count - 1;
		while (Left <= Right)
		{
			const int32 Bin = FMath::Clamp(
				(int32)((Triangles[Order[Left]].Centroid[BestAxis] - AxisMin) * Scale),
				0, GSahNumBins - 1);
			if (Bin <= BestBin)
			{
				++Left;
			}
			else
			{
				Swap(Order[Left], Order[Right]);
				--Right;
			}
		}
		const int32 LeftCount = Left - Start;
		return (LeftCount <= 0 || LeftCount >= Count) ? Count / 2 : LeftCount;
	}

	static void BuildFreeSahNode(
		TArray<FSahTriangle>& Triangles,
		TArray<uint32>& Order,
		int32 Start,
		int32 Count,
		uint32 LeafSize,
		uint32 Arity,
		TArray<FSahBuildNode>& OutTree,
		int32 NodeIndex)
	{
		if ((uint32)Count <= LeafSize)
		{
			FSahBuildNode& Leaf = OutTree[NodeIndex];
			Leaf.bLeaf = true;
			Leaf.FirstTriangle = (uint32)Start;
			Leaf.TriangleCount = (uint32)Count;
			Leaf.Height = 0;
			return;
		}
		TArray<FIntPoint, TInlineAllocator<8>> Ranges;
		Ranges.Add(FIntPoint(Start, Count));
		while (Ranges.Num() < (int32)Arity)
		{
			int32 Worst = INDEX_NONE;
			double WorstCost = 0.0;
			for (int32 Index = 0; Index < Ranges.Num(); ++Index)
			{
				if ((uint32)Ranges[Index].Y <= LeafSize || Ranges[Index].Y < 2)
				{
					continue;
				}
				FVector3f RangeMin, RangeMax;
				BoundsOfRange(Triangles, Order, Ranges[Index].X, Ranges[Index].Y, RangeMin, RangeMax);
				const double Cost = HalfSurfaceArea(RangeMin, RangeMax) * (double)Ranges[Index].Y;
				if (Cost > WorstCost)
				{
					WorstCost = Cost;
					Worst = Index;
				}
			}
			if (Worst == INDEX_NONE)
			{
				break;
			}
			const FIntPoint Range = Ranges[Worst];
			const int32 LeftCount = SplitRangeFreeSAH(Triangles, Order, Range.X, Range.Y);
			if (LeftCount <= 0 || LeftCount >= Range.Y)
			{
				break;
			}
			Ranges[Worst] = FIntPoint(Range.X, LeftCount);
			Ranges.Insert(FIntPoint(Range.X + LeftCount, Range.Y - LeftCount), Worst + 1);
		}
		if (Ranges.Num() < 2)
		{
			FSahBuildNode& Leaf = OutTree[NodeIndex];
			Leaf.bLeaf = true;
			Leaf.FirstTriangle = (uint32)Start;
			Leaf.TriangleCount = (uint32)FMath::Min<int32>(Count, GMaxLeafSize);
			Leaf.Height = 0;
			return;
		}
		// Widest child first. The traversal tests a node's children in slot
		// order and a shadow ray stops at its first hit, so the child most
		// likely to hold one is worth testing before its siblings. Surface area
		// is that likelihood: a wider box catches more directions. Ordering
		// only, so the tree's shape, its triangle ranges and the memory they
		// live in are all unchanged.
		{
			// Area per range once, not inside the comparison, so this costs one
			// pass over the node's triangles rather than one per compare.
			struct FChildOrder
			{
				FIntPoint Range = FIntPoint(0, 0);
				double Area = 0.0;
			};
			TArray<FChildOrder, TInlineAllocator<8>> ChildOrder;
			ChildOrder.Reserve(Ranges.Num());
			for (const FIntPoint& Range : Ranges)
			{
				FVector3f RangeMin, RangeMax;
				BoundsOfRange(Triangles, Order, Range.X, Range.Y, RangeMin, RangeMax);
				ChildOrder.Add({ Range, HalfSurfaceArea(RangeMin, RangeMax) });
			}
			ChildOrder.Sort(
				[](const FChildOrder& A, const FChildOrder& B)
				{
					// Ties keep the lower range first, so a build is repeatable.
					return A.Area != B.Area ? A.Area > B.Area : A.Range.X < B.Range.X;
				});
			for (int32 Child = 0; Child < Ranges.Num(); ++Child)
			{
				Ranges[Child] = ChildOrder[Child].Range;
			}
		}
		const int32 FirstChild = OutTree.Num();
		OutTree.AddDefaulted(Ranges.Num());
		for (int32 Child = 0; Child < Ranges.Num(); ++Child)
		{
			BuildFreeSahNode(
				Triangles, Order, Ranges[Child].X, Ranges[Child].Y,
				LeafSize, Arity, OutTree, FirstChild + Child);
		}
		int32 MaxChildHeight = 0;
		for (int32 Child = 0; Child < Ranges.Num(); ++Child)
		{
			MaxChildHeight = FMath::Max(MaxChildHeight, OutTree[FirstChild + Child].Height);
		}
		OutTree[NodeIndex].bLeaf = false;
		OutTree[NodeIndex].FirstChild = (uint32)FirstChild;
		OutTree[NodeIndex].ChildCount = (uint32)Ranges.Num();
		OutTree[NodeIndex].Height = MaxChildHeight + 1;
	}

	static void BuildFreeSahTree(
		TArray<FSahTriangle>& Triangles,
		TArray<uint32>& Order,
		uint32 LeafSize,
		uint32 Arity,
		TArray<FSahBuildNode>& OutTree)
	{
		OutTree.Reset();
		if (Order.Num() == 0)
		{
			return;
		}
		OutTree.AddDefaulted(1);
		BuildFreeSahNode(Triangles, Order, 0, Order.Num(), LeafSize, Arity, OutTree, 0);
	}

	static void BuildUniformTree(
		uint32 TriangleCount, uint32 LeafSize, uint32 Arity, TArray<FSahBuildNode>& Out)
	{
		Out.Reset();
		if (TriangleCount == 0)
		{
			return;
		}
		const int32 LeafCount = (int32)FMath::DivideAndRoundUp(TriangleCount, LeafSize);
		for (int32 Leaf = 0; Leaf < LeafCount; ++Leaf)
		{
			const uint32 First = (uint32)Leaf * LeafSize;
			FSahBuildNode& Node = Out.AddDefaulted_GetRef();
			Node.bLeaf = true;
			Node.FirstTriangle = First;
			Node.TriangleCount = FMath::Min(LeafSize, TriangleCount - First);
			Node.Height = 0;
		}
		int32 PreviousStart = 0;
		int32 PreviousCount = LeafCount;
		int32 Height = 0;
		while (PreviousCount > 1)
		{
			const int32 ThisCount = (int32)FMath::DivideAndRoundUp((uint32)PreviousCount, Arity);
			const int32 ThisStart = Out.Num();
			++Height;
			for (int32 Node = 0; Node < ThisCount; ++Node)
			{
				const int32 FirstChild = Node * (int32)Arity;
				FSahBuildNode& Interior = Out.AddDefaulted_GetRef();
				Interior.bLeaf = false;
				Interior.FirstChild = (uint32)(PreviousStart + FirstChild);
				Interior.ChildCount =
					(uint32)FMath::Min((int32)Arity, PreviousCount - FirstChild);
				Interior.Height = Height;
			}
			PreviousStart = ThisStart;
			PreviousCount = ThisCount;
		}
	}

	static void EmitCollapsedNode(
		const TArray<FSahBuildNode>& In,
		int32 InIndex,
		TConstArrayView<uint32> CollapsedCount,
		TConstArrayView<uint32> CollapsedFirst,
		TArray<FSahBuildNode>& Out,
		int32 OutIndex)
	{
		if (CollapsedCount[InIndex] != 0)
		{
			FSahBuildNode& Leaf = Out[OutIndex];
			Leaf.bLeaf = true;
			Leaf.FirstTriangle = CollapsedFirst[InIndex];
			Leaf.TriangleCount = CollapsedCount[InIndex];
			Leaf.Height = 0;
			return;
		}
		const FSahBuildNode& Node = In[InIndex];
		const int32 FirstChild = Out.Num();
		Out.AddDefaulted((int32)Node.ChildCount);
		for (uint32 Child = 0; Child < Node.ChildCount; ++Child)
		{
			EmitCollapsedNode(
				In, (int32)Node.FirstChild + (int32)Child,
				CollapsedCount, CollapsedFirst,
				Out, FirstChild + (int32)Child);
		}
		int32 MaxChildHeight = 0;
		for (uint32 Child = 0; Child < Node.ChildCount; ++Child)
		{
			MaxChildHeight = FMath::Max(MaxChildHeight, Out[FirstChild + (int32)Child].Height);
		}
		Out[OutIndex].bLeaf = false;
		Out[OutIndex].FirstChild = (uint32)FirstChild;
		Out[OutIndex].ChildCount = Node.ChildCount;
		Out[OutIndex].Height = MaxChildHeight + 1;
	}

	static int32 CollapseNarrowNodes(
		TArray<FSahBuildNode>& Tree,
		int32 RootIndex,
		uint32 Budget)
	{
		if (Tree.Num() <= 1 || Budget <= 1)
		{
			return RootIndex;
		}
		const int32 Num = Tree.Num();
		TArray<uint32> CollapsedCount;
		TArray<uint32> CollapsedFirst;
		CollapsedCount.Init(0, Num);
		CollapsedFirst.Init(0, Num);
		int32 MaxNodeHeight = 0;
		for (int32 Index = 0; Index < Num; ++Index)
		{
			MaxNodeHeight = FMath::Max(MaxNodeHeight, Tree[Index].Height);
		}
		TArray<TArray<int32>> ByHeight;
		ByHeight.SetNum(MaxNodeHeight + 1);
		for (int32 Index = 0; Index < Num; ++Index)
		{
			ByHeight[FMath::Clamp(Tree[Index].Height, 0, MaxNodeHeight)].Add(Index);
		}
		bool bAnyCollapsed = false;
		for (int32 Height = 0; Height <= MaxNodeHeight; ++Height)
		{
			for (int32 Index : ByHeight[Height])
			{
				const FSahBuildNode& Node = Tree[Index];
				if (Node.bLeaf)
				{
					CollapsedCount[Index] = Node.TriangleCount;
					CollapsedFirst[Index] = Node.FirstTriangle;
					continue;
				}
				uint32 Total = 0;
				uint32 First = MAX_uint32;
				bool bEveryChildIsALeaf = Node.ChildCount > 0;
				for (uint32 Child = 0; Child < Node.ChildCount; ++Child)
				{
					const int32 ChildIndex = (int32)Node.FirstChild + (int32)Child;
					if (ChildIndex < 0 || ChildIndex >= Num || CollapsedCount[ChildIndex] == 0)
					{
						bEveryChildIsALeaf = false;
						break;
					}
					Total += CollapsedCount[ChildIndex];
					First = FMath::Min(First, CollapsedFirst[ChildIndex]);
				}
				if (bEveryChildIsALeaf && Total <= Budget)
				{
					CollapsedCount[Index] = Total;
					CollapsedFirst[Index] = First;
					bAnyCollapsed = true;
				}
			}
		}
		if (!bAnyCollapsed)
		{
			return RootIndex;
		}
		TArray<FSahBuildNode> Collapsed;
		Collapsed.Reserve(Num);
		Collapsed.AddDefaulted(1);
		EmitCollapsedNode(Tree, RootIndex, CollapsedCount, CollapsedFirst, Collapsed, 0);
		Tree = MoveTemp(Collapsed);
		return 0;
	}

	struct FPendingSahBuild
	{
		int32 Slot = INDEX_NONE;
		uint32 TriangleCount = 0;
		TArray<float> Triangles;
		TArray<uint32> Order;
		TArray<FSahBuildNode> Tree;
		FBox ExactBounds = FBox(ForceInit);
		FTriangleShapeProbe Probe;
		UE::Tasks::FTask Task;
	};

	static TArray<TUniquePtr<FPendingSahBuild>> GPendingSahBuilds;

	static bool IsSahBuildPending(int32 Slot)
	{
		for (const TUniquePtr<FPendingSahBuild>& Build : GPendingSahBuilds)
		{
			if (Build->Slot == Slot)
			{
				return true;
			}
		}
		return false;
	}

	static void DrainSahBuilds()
	{
		for (TUniquePtr<FPendingSahBuild>& Build : GPendingSahBuilds)
		{
			if (Build->Task.IsValid())
			{
				Build->Task.Wait();
			}
		}
		GPendingSahBuilds.Reset();
	}

	struct FPendingLayoutBuild
	{
		TArray<uint32> TriangleOffsets;
		TArray<uint32> TriangleCounts;
		TArray<TArray<FSahBuildNode>> FreeTrees;
		uint32 LeafSize = 1;
		uint32 Arity = 4;
		uint32 CollapseBudget = 1;
		int32 FreeTreeCount = 0;
		bool bTreeStats = false;
		FNodeLayout Out;
		UE::Tasks::FTask Task;
	};

	static TUniquePtr<FPendingLayoutBuild> GPendingLayout;

	static void DrainLayoutBuild()
	{
		if (GPendingLayout.IsValid())
		{
			if (GPendingLayout->Task.IsValid())
			{
				GPendingLayout->Task.Wait();
			}
			GPendingLayout.Reset();
		}
	}

	static void BuildNodeLayout(
		TConstArrayView<uint32> TriangleOffsets,
		TConstArrayView<uint32> TriangleCounts,
		uint32 LeafSize,
		uint32 Arity,
		TConstArrayView<TArray<FSahBuildNode>> FreeTrees,
		uint32 CollapseBudget,
		bool bTreeStats,
		TArray<TArray<FSahBuildNode>>* MemoTrees,
		TArray<int32>* MemoRootLocal,
		TArray<uint64>* MemoKey,
		TConstArrayView<uint32> TreeSerials,
		FNodeLayout& Out)
	{
		const int32 NumSources = TriangleCounts.Num();
		static const TArray<FSahBuildNode> EmptyTree;
		TArray<TArray<FSahBuildNode>> LocalTrees;
		LocalTrees.SetNum(MemoTrees != nullptr ? 0 : NumSources);
		TArray<const TArray<FSahBuildNode>*> SourceTreePtr;
		SourceTreePtr.Init(nullptr, NumSources);
		auto SourceTree = [&SourceTreePtr](int32 Index) -> const TArray<FSahBuildNode>&
		{
			return SourceTreePtr[Index] != nullptr ? *SourceTreePtr[Index] : EmptyTree;
		};
		TArray<int32> SourceBase;
		SourceBase.SetNumZeroed(NumSources);
		TArray<int32> SourceRootLocal;
		SourceRootLocal.Init(INDEX_NONE, NumSources);
		TArray<bool> bSourceUsedFree;
		bSourceUsedFree.Init(false, NumSources);
		CollapseBudget = (uint32)FMath::Clamp((int32)CollapseBudget, 1, GMaxLeafSize);
		int32 TotalNodes = 0;
		int32 MaxHeight = 0;
		for (int32 SourceIndex = 0; SourceIndex < NumSources; ++SourceIndex)
		{
			if (TriangleCounts[SourceIndex] == 0)
			{
				continue;
			}
			const bool bHaveFree =
				FreeTrees.IsValidIndex(SourceIndex)
				&& FreeTrees[SourceIndex].Num() > 0;
			bSourceUsedFree[SourceIndex] = bHaveFree;
			const uint64 SlotKey = MemoKey != nullptr
				? (uint64)HashCombine(
					HashCombine(
						HashCombine(GetTypeHash(TriangleCounts[SourceIndex]), GetTypeHash(LeafSize)),
						HashCombine(GetTypeHash(Arity), GetTypeHash(CollapseBudget))),
					HashCombine(
						GetTypeHash(bHaveFree),
						GetTypeHash(TreeSerials.IsValidIndex(SourceIndex)
							? TreeSerials[SourceIndex] : 0u))) | 0x100000000ull
				: 0ull;
			const bool bMemoHit =
				MemoKey != nullptr
				&& MemoTrees->IsValidIndex(SourceIndex)
				&& (*MemoKey)[SourceIndex] == SlotKey
				&& (*MemoTrees)[SourceIndex].Num() > 0;
			TArray<FSahBuildNode>& Dest = MemoTrees != nullptr
				? (*MemoTrees)[SourceIndex]
				: LocalTrees[SourceIndex];
			if (!bMemoHit)
			{
				if (bHaveFree)
				{
					Dest = FreeTrees[SourceIndex];
					SourceRootLocal[SourceIndex] = 0;
				}
				else
				{
					BuildUniformTree(TriangleCounts[SourceIndex], LeafSize, Arity, Dest);
					SourceRootLocal[SourceIndex] = Dest.Num() - 1;
				}
				if (CollapseBudget > 1)
				{
					SourceRootLocal[SourceIndex] =
						CollapseNarrowNodes(Dest, SourceRootLocal[SourceIndex], CollapseBudget);
				}
				if (MemoKey != nullptr)
				{
					(*MemoRootLocal)[SourceIndex] = SourceRootLocal[SourceIndex];
					(*MemoKey)[SourceIndex] = SlotKey;
				}
			}
			else
			{
				SourceRootLocal[SourceIndex] = (*MemoRootLocal)[SourceIndex];
			}
			SourceTreePtr[SourceIndex] = &Dest;
			SourceBase[SourceIndex] = TotalNodes;
			TotalNodes += Dest.Num();
			for (const FSahBuildNode& Node : Dest)
			{
				MaxHeight = FMath::Max(MaxHeight, Node.Height);
			}
		}
		if (bTreeStats)
		{
			int32 ChildHistogram[9] = { 0 };
			int32 InternalNodes = 0;
			int32 LeafNodes = 0;
			int32 Edges = 0;
			int32 LeafTriangles = 0;
			int32 WidestLeaf = 0;
			int32 FreeSources = 0;
			int32 UniformSources = 0;
			for (int32 SourceIndex = 0; SourceIndex < NumSources; ++SourceIndex)
			{
				if (SourceTree(SourceIndex).Num() == 0)
				{
					continue;
				}
				if (bSourceUsedFree[SourceIndex])
				{
					++FreeSources;
				}
				else
				{
					++UniformSources;
				}
				for (const FSahBuildNode& Node : SourceTree(SourceIndex))
				{
					if (Node.bLeaf)
					{
						++LeafNodes;
						LeafTriangles += (int32)Node.TriangleCount;
						WidestLeaf = FMath::Max(WidestLeaf, (int32)Node.TriangleCount);
					}
					else
					{
						++InternalNodes;
						Edges += (int32)Node.ChildCount;
						ChildHistogram[FMath::Clamp((int32)Node.ChildCount, 0, 8)] += 1;
					}
				}
			}
			const double Branching = InternalNodes > 0
				? (double)Edges / (double)InternalNodes : 0.0;
			const int32 IdealInternal = Arity > 1
				? (int32)FMath::DivideAndRoundUp(FMath::Max(LeafNodes - 1, 0), (int32)Arity - 1)
				: 0;
			const int32 IdealNodes = LeafNodes + IdealInternal;
			UE_LOG(LogTemp, Display,
				TEXT("SWRT.Trace TREE: %d nodes = %d internal + %d leaves")
				TEXT(" | branching %.3f of arity %u | children 2:%d 3:%d 4:%d 5:%d 6:%d 7:%d 8:%d")
				TEXT(" (degenerate 0:%d 1:%d)")
				TEXT(" | leaves hold %.2f tris mean, %d widest, leaf size %u, collapse %u")
				TEXT(" | at a true arity %u the same leaves would be %d nodes (%+.1f%%)")
				TEXT(" | builders: %d free, %d uniform | max height %d"),
				InternalNodes + LeafNodes, InternalNodes, LeafNodes,
				Branching, Arity,
				ChildHistogram[2], ChildHistogram[3], ChildHistogram[4], ChildHistogram[5],
				ChildHistogram[6], ChildHistogram[7], ChildHistogram[8],
				ChildHistogram[0], ChildHistogram[1],
				LeafNodes > 0 ? (double)LeafTriangles / (double)LeafNodes : 0.0,
				WidestLeaf, LeafSize, CollapseBudget,
				Arity, IdealNodes,
				IdealNodes > 0
					? 100.0 * ((double)(InternalNodes + LeafNodes) - (double)IdealNodes)
						/ (double)IdealNodes
					: 0.0,
				FreeSources, UniformSources, MaxHeight);
		}
		Out.Descs.SetNumUninitialized(TotalNodes);
		Out.NodeSources.SetNumUninitialized(TotalNodes);
		Out.Roots.Init(INDEX_NONE, NumSources);
		Out.SourceLevelSlices.SetNum(NumSources);
		for (int32 SourceIndex = 0; SourceIndex < NumSources; ++SourceIndex)
		{
			const TArray<FSahBuildNode>& Tree = SourceTree(SourceIndex);
			if (Tree.Num() == 0)
			{
				continue;
			}
			const int32 Base = SourceBase[SourceIndex];
			for (int32 Local = 0; Local < Tree.Num(); ++Local)
			{
				const FSahBuildNode& Node = Tree[Local];
				Out.NodeSources[Base + Local] = (uint32)SourceIndex;
				Out.Descs[Base + Local] = Node.bLeaf
					? PackNodeDesc(
						TriangleOffsets[SourceIndex] + Node.FirstTriangle,
						Node.TriangleCount,
						true)
					: PackNodeDesc(
						(uint32)(Base + (int32)Node.FirstChild),
						Node.ChildCount,
						false);
			}
			Out.Roots[SourceIndex] = Base + SourceRootLocal[SourceIndex];
		}
		{
			const int32 NumHeights = MaxHeight + 1;
			const int32 NumBuckets = NumHeights * NumSources;
			TArray<int32> BucketCounts;
			BucketCounts.SetNumZeroed(NumBuckets);
			for (int32 SourceIndex = 0; SourceIndex < NumSources; ++SourceIndex)
			{
				for (const FSahBuildNode& Node : SourceTree(SourceIndex))
				{
					++BucketCounts[Node.Height * NumSources + SourceIndex];
				}
			}
			TArray<int32> BucketStart;
			BucketStart.SetNumUninitialized(NumBuckets);
			int32 Running = 0;
			for (int32 Bucket = 0; Bucket < NumBuckets; ++Bucket)
			{
				BucketStart[Bucket] = Running;
				Running += BucketCounts[Bucket];
			}
			Out.LevelNodeIndices.SetNumUninitialized(Running);
			TArray<int32> Cursor = BucketStart;
			for (int32 SourceIndex = 0; SourceIndex < NumSources; ++SourceIndex)
			{
				const TArray<FSahBuildNode>& Tree = SourceTree(SourceIndex);
				const int32 Base = SourceBase[SourceIndex];
				for (int32 Local = 0; Local < Tree.Num(); ++Local)
				{
					const int32 Bucket = Tree[Local].Height * NumSources + SourceIndex;
					Out.LevelNodeIndices[Cursor[Bucket]++] = (uint32)(Base + Local);
				}
			}
			Out.Levels.SetNumUninitialized(NumHeights);
			int32 LevelStart = 0;
			for (int32 Height = 0; Height < NumHeights; ++Height)
			{
				int32 LevelCount = 0;
				for (int32 SourceIndex = 0; SourceIndex < NumSources; ++SourceIndex)
				{
					const int32 Bucket = Height * NumSources + SourceIndex;
					LevelCount += BucketCounts[Bucket];
					if (SourceTree(SourceIndex).Num() > 0)
					{
						Out.SourceLevelSlices[SourceIndex].Add(
							FIntPoint(BucketStart[Bucket], BucketCounts[Bucket]));
					}
				}
				Out.Levels[Height] = FIntPoint(LevelStart, LevelCount);
				LevelStart += LevelCount;
			}
		}
		Out.NumBlasLevels = Out.Levels.Num();
		BuildTopLevel(Arity, Out);
	}

	static void BuildTopLevel(uint32 Arity, FNodeLayout& Out)
	{
		TArray<int32> Instances;
		Instances.Reserve(Out.Roots.Num());
		Out.InstanceSlots.Reset();
		for (int32 Slot = 0; Slot < Out.Roots.Num(); ++Slot)
		{
			if (Out.Roots[Slot] != INDEX_NONE)
			{
				Instances.Add(Out.Roots[Slot]);
				Out.InstanceSlots.Add(Slot);
			}
		}
		if (Instances.Num() == 0)
		{
			return;
		}
		const int32 InstanceStart = Out.Descs.Num();
		const int32 InstanceListStart = Out.LevelNodeIndices.Num();
		for (int32 Index = 0; Index < Instances.Num(); ++Index)
		{
			Out.Descs.Add(PackNodeDesc((uint32)Instances[Index], 1, false));
			Out.NodeSources.Add(0);
			Out.LevelNodeIndices.Add((uint32)(InstanceStart + Index));
		}
		Out.Levels.Add(FIntPoint(InstanceListStart, Instances.Num()));
		int32 PreviousStart = InstanceStart;
		int32 PreviousCount = Instances.Num();
		while (PreviousCount > 1)
		{
			const int32 Count = (int32)FMath::DivideAndRoundUp((uint32)PreviousCount, Arity);
			const int32 Start = Out.Descs.Num();
			const int32 ListStart = Out.LevelNodeIndices.Num();
			for (int32 Node = 0; Node < Count; ++Node)
			{
				const int32 FirstChild = Node * (int32)Arity;
				Out.Descs.Add(PackNodeDesc(
					(uint32)(PreviousStart + FirstChild),
					(uint32)FMath::Min((int32)Arity, PreviousCount - FirstChild),
					false));
				Out.NodeSources.Add(0);
				Out.LevelNodeIndices.Add((uint32)(Start + Node));
			}
			Out.Levels.Add(FIntPoint(ListStart, Count));
			PreviousStart = Start;
			PreviousCount = Count;
		}
		Out.TlasRoot = PreviousStart;
	}

	/*=========================================================================
		REFIT

		The tree's SHAPE is built on the CPU and reused; its BOUNDS are
		recomputed on the GPU every frame, bottom up, one dispatch a level. That
		is what lets a skinned source keep its tree while its vertices move.
	=========================================================================*/

	class FSWRTBVHBoundsCS : public FGlobalShader
	{
		DECLARE_GLOBAL_SHADER(FSWRTBVHBoundsCS);
		SHADER_USE_PARAMETER_STRUCT(FSWRTBVHBoundsCS, FGlobalShader);
		class FLeafLevelDim : SHADER_PERMUTATION_BOOL("SWRT_BVH_LEAF_LEVEL");
		class FPerSourceFrameDim : SHADER_PERMUTATION_BOOL("SWRT_BVH_PER_SOURCE_FRAME");
		class FInstanceLevelDim : SHADER_PERMUTATION_BOOL("SWRT_BVH_INSTANCE_LEVEL");
		using FPermutationDomain =
			TShaderPermutationDomain<
				FLeafLevelDim, FPerSourceFrameDim, FInstanceLevelDim>;
		BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
			SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<float>, SWRTBvhTriangles)
			SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, SWRTBvhDescs)
			SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, SWRTBvhLevelNodes)
			SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, SWRTBvhInstanceVisible)
			SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, SWRTBvhNodeSources)
			SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<float4>, SWRTBvhFrames)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint4>, SWRTBvhNodes)
			SHADER_PARAMETER(FVector3f, SWRTBvhSceneMin)
			SHADER_PARAMETER(FVector3f, SWRTBvhQuantScale)
			SHADER_PARAMETER(uint32, SWRTBvhNodeOffset)
			SHADER_PARAMETER(uint32, SWRTBvhNodeCount)
		END_SHADER_PARAMETER_STRUCT()
		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
		{
			return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
		}
		static constexpr uint32 ThreadGroupSize = 64;
	};

	IMPLEMENT_GLOBAL_SHADER(FSWRTBVHBoundsCS, "/Plugins/PrototypeSWRTShadows/SWRTBVHBoundsCS.usf", "MainCS", SF_Compute);

	static FRDGBufferRef BuildNodeBounds(
		FRDGBuilder& GraphBuilder,
		const FViewInfo& View,
		FRDGBufferRef TriangleBuffer,
		const FNodeLayout& Layout,
		const FVector3f& SceneMin,
		const FVector3f& QuantScale,
		TConstArrayView<FVector4f> SourceFrames,
		TConstArrayView<uint32> InstanceVisible,
		int32 InstanceLevel,
		TConstArrayView<uint32> RefitNodeIndices,
		TConstArrayView<FIntPoint> RefitLevels,
		FTriangleCache* Cache,
		bool bShapeUnchanged)
	{
		const int32 NumNodes = Layout.Descs.Num();
		if (NumNodes == 0)
		{
			return nullptr;
		}
		const bool bReuseShape = Cache != nullptr && bShapeUnchanged
			&& Cache->Descs.IsValid() && Cache->NodeSources.IsValid();
		FRDGBufferRef DescBuffer = bReuseShape
			? GraphBuilder.RegisterExternalBuffer(Cache->Descs)
			: CreateStructuredBuffer(
				GraphBuilder, TEXT("SWRT.TraceNodeDescs"),
				sizeof(uint32), NumNodes, Layout.Descs.GetData(), NumNodes * sizeof(uint32));
		FRDGBufferRef LevelNodeBuffer = CreateStructuredBuffer(
			GraphBuilder, TEXT("SWRT.TraceLevelNodes"),
			sizeof(uint32), RefitNodeIndices.Num(), RefitNodeIndices.GetData(),
			RefitNodeIndices.Num() * sizeof(uint32));
		const bool bPerSourceFrame = SourceFrames.Num() > 0;
		const FVector4f DefaultFrame[2] = { FVector4f(ForceInitToZero), FVector4f(ForceInitToZero) };
		FRDGBufferRef FrameBuffer = CreateStructuredBuffer(
			GraphBuilder, TEXT("SWRT.TraceFrames"),
			sizeof(FVector4f),
			bPerSourceFrame ? SourceFrames.Num() : 2,
			bPerSourceFrame ? (const void*)SourceFrames.GetData() : (const void*)DefaultFrame,
			(bPerSourceFrame ? SourceFrames.Num() : 2) * sizeof(FVector4f));
		FRDGBufferRef NodeSourceBuffer = bReuseShape
			? GraphBuilder.RegisterExternalBuffer(Cache->NodeSources)
			: CreateStructuredBuffer(
				GraphBuilder, TEXT("SWRT.TraceNodeSources"),
				sizeof(uint32), Layout.NodeSources.Num(), Layout.NodeSources.GetData(),
				Layout.NodeSources.Num() * sizeof(uint32));
		const bool bReuseNodes = Cache != nullptr && bShapeUnchanged && Cache->Nodes.IsValid();
		FRDGBufferRef NodeBuffer = bReuseNodes
			? GraphBuilder.RegisterExternalBuffer(Cache->Nodes)
			: GraphBuilder.CreateBuffer(
				FRDGBufferDesc::CreateBufferDesc(sizeof(FUintVector4), NumNodes),
				TEXT("SWRT.TraceNodes"));
		FRDGBufferUAVRef NodeUAV = GraphBuilder.CreateUAV(NodeBuffer, PF_R32G32B32A32_UINT);
		FRDGBufferSRVRef DescSRV = GraphBuilder.CreateSRV(DescBuffer, PF_R32_UINT);
		const uint32 VisibleFallback = 1;
		FRDGBufferRef VisibleBuffer = CreateStructuredBuffer(
			GraphBuilder, TEXT("SWRT.TraceInstanceVisible"),
			sizeof(uint32),
			FMath::Max(InstanceVisible.Num(), 1),
			InstanceVisible.Num() > 0 ? (const void*)InstanceVisible.GetData() : (const void*)&VisibleFallback,
			FMath::Max(InstanceVisible.Num(), 1) * sizeof(uint32));
		FRDGBufferSRVRef LevelNodeSRV = GraphBuilder.CreateSRV(LevelNodeBuffer, PF_R32_UINT);
		FRDGBufferSRVRef VisibleSRV = GraphBuilder.CreateSRV(VisibleBuffer, PF_R32_UINT);
		FRDGBufferSRVRef FrameSRV = GraphBuilder.CreateSRV(FrameBuffer, PF_A32B32G32R32F);
		FRDGBufferSRVRef NodeSourceSRV = GraphBuilder.CreateSRV(NodeSourceBuffer, PF_R32_UINT);
		FRDGBufferSRVRef TriangleSRV = GraphBuilder.CreateSRV(
			TriangleBuffer, PF_A32B32G32R32F);

		// One scope over every level, so a capture reports the whole refit as a
		// single number instead of one row per level.
		RDG_EVENT_SCOPE(GraphBuilder, "SWRT.BVHRefit %d levels, %d nodes",
			RefitLevels.Num(), RefitNodeIndices.Num());

		for (int32 Level = 0; Level < RefitLevels.Num(); ++Level)
		{
			const FIntPoint Range = RefitLevels[Level];
			if (Range.Y <= 0)
			{
				continue;
			}
			FSWRTBVHBoundsCS::FPermutationDomain PermutationVector;
			PermutationVector.Set<FSWRTBVHBoundsCS::FLeafLevelDim>(Level == 0);
			PermutationVector.Set<FSWRTBVHBoundsCS::FPerSourceFrameDim>(bPerSourceFrame);
			PermutationVector.Set<FSWRTBVHBoundsCS::FInstanceLevelDim>(Level == InstanceLevel);
			TShaderMapRef<FSWRTBVHBoundsCS> ComputeShader(
				GetGlobalShaderMap(View.GetFeatureLevel()), PermutationVector);
			FSWRTBVHBoundsCS::FParameters* Parameters =
				GraphBuilder.AllocParameters<FSWRTBVHBoundsCS::FParameters>();
			Parameters->SWRTBvhTriangles = TriangleSRV;
			Parameters->SWRTBvhDescs = DescSRV;
			Parameters->SWRTBvhLevelNodes = LevelNodeSRV;
			Parameters->SWRTBvhInstanceVisible = VisibleSRV;
			Parameters->SWRTBvhNodeSources = NodeSourceSRV;
			Parameters->SWRTBvhFrames = FrameSRV;
			Parameters->SWRTBvhNodes = NodeUAV;
			Parameters->SWRTBvhSceneMin = SceneMin;
			Parameters->SWRTBvhQuantScale = QuantScale;
			Parameters->SWRTBvhNodeOffset = (uint32)Range.X;
			Parameters->SWRTBvhNodeCount = (uint32)Range.Y;
			FComputeShaderUtils::AddPass(
				GraphBuilder,
				RDG_EVENT_NAME("SWRT.BVHBounds level %d, %d nodes", Level, Range.Y),
				ComputeShader,
				Parameters,
				FIntVector(FMath::DivideAndRoundUp((uint32)Range.Y, FSWRTBVHBoundsCS::ThreadGroupSize), 1, 1));
		}
		if (Cache != nullptr)
		{
			if (!bReuseNodes)
			{
				GraphBuilder.QueueBufferExtraction(NodeBuffer, &Cache->Nodes, ERHIAccess::SRVCompute);
			}
			if (!bReuseShape)
			{
				GraphBuilder.QueueBufferExtraction(DescBuffer, &Cache->Descs, ERHIAccess::SRVCompute);
				GraphBuilder.QueueBufferExtraction(NodeSourceBuffer, &Cache->NodeSources, ERHIAccess::SRVCompute);
			}
		}
		return NodeBuffer;
	}

	/*=========================================================================
		THE TRACE
	=========================================================================*/

	class FSWRTTraceCS : public FGlobalShader
	{
		DECLARE_GLOBAL_SHADER(FSWRTTraceCS);
		SHADER_USE_PARAMETER_STRUCT(FSWRTTraceCS, FGlobalShader);
		/**
		 * The counters, their atomics and the debug views exist only in the
		 * stats permutation, which is what keeps every one of them out of the
		 * shader that actually ships.
		 */
		class FStatsDim : SHADER_PERMUTATION_BOOL("SWRT_STATS");
		using FPermutationDomain = TShaderPermutationDomain<FStatsDim>;

		static void ModifyCompilationEnvironment(
			const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
		{
			FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
			OutEnvironment.SetDefine(TEXT("SHADING_PATH_DEFERRED"), 1);
		}

		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
		{
			return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
		}

		BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
			SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
			SHADER_PARAMETER_STRUCT_INCLUDE(FSceneTextureShaderParameters, SceneTextures)
			SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<float>, SWRTTriangles)
			SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<float4>, SWRTCasters)
			SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, SWRTRoots)
			SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<float4>, SWRTInstances)
			SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<float4>, SWRTInstanceTransforms)
			SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint4>, SWRTInstanceNodes)
			SHADER_PARAMETER(uint32, SWRTInstanceRoot)
			SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint4>, SWRTNodes)
			SHADER_PARAMETER(FVector3f, SWRTSceneMin)
			SHADER_PARAMETER(FVector3f, SWRTQuantScale)
			SHADER_PARAMETER(FVector3f, SWRTTriangleSpaceOffset)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, SWRTStats)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, SWRTCasterHitsOut)
			SHADER_PARAMETER(uint32, SWRTCasterCount)
			SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float>, SWRTSceneDepth)
			SHADER_PARAMETER_RDG_TEXTURE_UAV(RWTexture2D<float>, SWRTOutMask)
			SHADER_PARAMETER(FVector3f, SWRTLightPosition)
			SHADER_PARAMETER(float, SWRTLightRadius)
			SHADER_PARAMETER(FVector3f, SWRTLightDirection)
			SHADER_PARAMETER(float, SWRTCosOuterCone)
			SHADER_PARAMETER(FUintVector2, SWRTPixelMin)
			SHADER_PARAMETER(FUintVector2, SWRTPixelMax)
			SHADER_PARAMETER(FUintVector2, SWRTDispatchGrid)
			SHADER_PARAMETER(FUintVector2, SWRTTileGrid)
			SHADER_PARAMETER(float, SWRTHeatmapScale)
			SHADER_PARAMETER(uint32, SWRTDebugView)
			SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, SWRTTileInstances)
			SHADER_PARAMETER(uint32, SWRTTileInstanceStride)
			SHADER_PARAMETER(uint32, SWRTSwizzleTileWidth)
			SHADER_PARAMETER(uint32, SWRTSwizzlePerfectTileCTAs)
			SHADER_PARAMETER(uint32, SWRTSwizzleTotalPerfectCTAs)
			SHADER_PARAMETER(uint32, SWRTSwizzleLastTileWidth)
			SHADER_PARAMETER(FUintVector3, SWRTSwizzleMagic)
			SHADER_PARAMETER(FUintVector3, SWRTSwizzleMagicShift)
			SHADER_PARAMETER(float, SWRTRayStartBias)
			SHADER_PARAMETER(float, SWRTRayStartSlope)
		END_SHADER_PARAMETER_STRUCT()
		/** Four warps, two 8x8 tiles a group. */
		static FUintVector2 GetGroupSize()
		{
			return FUintVector2(8, 16);
		}
		struct FSwizzleConstants
		{
			uint32 PerfectTileCTAs = 1;
			uint32 TotalPerfectCTAs = 0;
			uint32 LastTileWidth = 1;
			FUintVector3 Magic = FUintVector3(0, 0, 0);
			FUintVector3 MagicShift = FUintVector3(0, 0, 0);
		};
		static bool MakeDivideMagic(uint32 D, uint64 MaxN, uint32& OutMagic, uint32& OutShift)
		{
			OutMagic = 0;
			OutShift = 0;
			if (D <= 1)
			{
				return false;
			}
			for (uint32 Shift = 1; Shift < 32; ++Shift)
			{
				const uint64 Pow = 1ull << Shift;
				if (Pow <= MaxN * (uint64)(D - 1))
				{
					continue;
				}
				const uint64 Magic = (Pow + D - 1) / D;
				if (Magic > 0xFFFFFFFFull || Magic * MaxN >= (1ull << 32))
				{
					break;
				}
				OutMagic = (uint32)Magic;
				OutShift = Shift;
				return true;
			}
			return false;
		}
		static FSwizzleConstants MakeSwizzleConstants(FUintVector2 Grid, uint32 TileWidth)
		{
			FSwizzleConstants Out;
			if (Grid.X == 0 || Grid.Y == 0 || TileWidth == 0)
			{
				return Out;
			}
			const uint64 TotalCTAs = (uint64)Grid.X * (uint64)Grid.Y;
			Out.PerfectTileCTAs = TileWidth * Grid.Y;
			Out.TotalPerfectCTAs = (Grid.X / TileWidth) * Out.PerfectTileCTAs;
			Out.LastTileWidth = FMath::Max(1u, Grid.X % TileWidth);
			MakeDivideMagic(Out.PerfectTileCTAs, TotalCTAs, Out.Magic.X, Out.MagicShift.X);
			MakeDivideMagic(TileWidth, Out.PerfectTileCTAs, Out.Magic.Y, Out.MagicShift.Y);
			MakeDivideMagic(Out.LastTileWidth, Out.PerfectTileCTAs, Out.Magic.Z, Out.MagicShift.Z);
			return Out;
		}
	};

	IMPLEMENT_GLOBAL_SHADER(FSWRTTraceCS, "/Plugins/PrototypeSWRTShadows/SWRTTraceCS.usf", "MainCS", SF_Compute);

	class FSWRTTileBoundsCS : public FGlobalShader
	{
		DECLARE_GLOBAL_SHADER(FSWRTTileBoundsCS);
		SHADER_USE_PARAMETER_STRUCT(FSWRTTileBoundsCS, FGlobalShader);
		class FExactGridDim : SHADER_PERMUTATION_BOOL("SWRT_TB_EXACT_GRID");
		using FPermutationDomain = TShaderPermutationDomain<FExactGridDim>;

		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
		{
			return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
		}
		BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
			SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, View)
			SHADER_PARAMETER_RDG_TEXTURE(Texture2D<float>, SWRTTbSceneDepth)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<float4>, SWRTTbOutBounds)
			SHADER_PARAMETER(FUintVector2, SWRTTbPixelMin)
			SHADER_PARAMETER(FUintVector2, SWRTTbPixelMax)
			SHADER_PARAMETER(FUintVector2, SWRTTbTileGrid)
			SHADER_PARAMETER(FVector3f, SWRTTbLightPosition)
			SHADER_PARAMETER(float, SWRTTbLightRadius)
			SHADER_PARAMETER(FVector3f, SWRTTbLightDirection)
			SHADER_PARAMETER(float, SWRTTbCosOuterCone)
		END_SHADER_PARAMETER_STRUCT()
	};

	IMPLEMENT_GLOBAL_SHADER(FSWRTTileBoundsCS, "/Plugins/PrototypeSWRTShadows/SWRTTileBoundsCS.usf", "MainCS", SF_Compute);

	class FSWRTTileInstanceCS : public FGlobalShader
	{
		DECLARE_GLOBAL_SHADER(FSWRTTileInstanceCS);
		SHADER_USE_PARAMETER_STRUCT(FSWRTTileInstanceCS, FGlobalShader);
		static constexpr uint32 ThreadGroupSize = 64;
		BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
			SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint4>, SWRTTiNodes)
			SHADER_PARAMETER(uint32, SWRTTiInstanceRoot)
			SHADER_PARAMETER(uint32, SWRTTiCasterCount)
			SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<float4>, SWRTTiBounds)
			SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, SWRTTiOutInstances)
			SHADER_PARAMETER(uint32, SWRTTiStride)
			SHADER_PARAMETER(FUintVector2, SWRTTiTileGrid)
			SHADER_PARAMETER(FVector3f, SWRTTiSceneMin)
			SHADER_PARAMETER(FVector3f, SWRTTiQuantScale)
			SHADER_PARAMETER(FVector3f, SWRTTiLightPosition)
		END_SHADER_PARAMETER_STRUCT()
		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
		{
			return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
		}
	};

	IMPLEMENT_GLOBAL_SHADER(FSWRTTileInstanceCS, "/Plugins/PrototypeSWRTShadows/SWRTTileInstanceCS.usf", "MainCS", SF_Compute);

	/*=========================================================================
		ONE FRAME

		Gather, extract, build what the budget allows, refit, list, dispatch.
		Returns null when there is nothing to trace, which the caller must read
		as no shadow rather than as everything lit.
	=========================================================================*/

	FRDGTextureRef TickSoftwareTrace(
		FRDGBuilder& GraphBuilder,
		const FViewInfo& View,
		FRDGTextureRef SceneDepthTexture,
		TConstArrayView<const FPrimitiveSceneProxy*> Casters,
		const SWRTTrace::FTraceLight& Light)
	{
		if (Casters.Num() == 0 || SceneDepthTexture == nullptr)
		{
			return nullptr;
		}
		static uint32 GTickFrame = 0;
		static uint32 GTickCallIndex = 0;
		static uint32 GTickCallsLastFrame = 1;
		if (GTickFrame != GFrameNumberRenderThread)
		{
			GTickCallsLastFrame = FMath::Max(GTickCallIndex, 1u);
			GTickFrame = GFrameNumberRenderThread;
			GTickCallIndex = 0;
		}
		const uint32 TickCallIndex = GTickCallIndex++;
		const bool bReadbackTurn =
			TickCallIndex == (GFrameNumberRenderThread % GTickCallsLastFrame);
		static uint32 LastBuildFrame = 0xFFFFFFFFu;
		static uint32 BuildsThisFrame = 0;
		if (GFrameNumberRenderThread != LastBuildFrame)
		{
			LastBuildFrame = GFrameNumberRenderThread;
			BuildsThisFrame = 0;
		}
		++BuildsThisFrame;
		FGatheredGeometry Gathered;
		struct FStageTimes
		{
			double Gather = 0.0, Dedup = 0.0, Slots = 0.0, Layout = 0.0;
			double RefitList = 0.0, Instances = 0.0, Total = 0.0;
			bool bRebuiltLayout = false;
		} Timing;
		const double TickStart = FPlatformTime::Seconds();
		double StageStart = TickStart;
		GatherCasterGeometry(GraphBuilder, Casters, Gathered);
		Timing.Gather = FPlatformTime::Seconds() - StageStart;
		StageStart = FPlatformTime::Seconds();
		// Two placements of one mesh collapse to one source here, which is what
		// lets them share a copy of the triangles and a tree.
		{
			TArray<FTriangleSource> UniqueSources;
			TMap<FCachedSourceKey, int32> GeometryLookup;
			UniqueSources.Reserve(Gathered.Sources.Num());
			Gathered.Instances.Reserve(Gathered.Sources.Num());
			Gathered.TotalTriangles = 0;
			for (const FTriangleSource& Source : Gathered.Sources)
			{
				FCachedSourceKey Key;
				Key.PositionSRV = Source.PositionSRV;
				Key.IndexSRV = Source.IndexSRV;
				Key.IndexBuffer = Source.IndexBuffer;
				Key.FirstIndex = Source.FirstIndex;
				Key.NumPrimitives = Source.NumPrimitives;
				Key.Instance = Source.bSkinned ? Source.Instance : nullptr;
				int32 SourceIndex = INDEX_NONE;
				if (const int32* Existing = GeometryLookup.Find(Key))
				{
					SourceIndex = *Existing;
				}
				else
				{
					SourceIndex = UniqueSources.Add(Source);
					GeometryLookup.Add(Key, SourceIndex);
					Gathered.TotalTriangles += Source.NumPrimitives;
				}
				FGatheredInstance& Instance = Gathered.Instances.AddDefaulted_GetRef();
				Instance.SourceIndex = SourceIndex;
				Instance.LocalToWorld = Source.LocalToWorld;
				Instance.WorldBounds = Source.WorldBounds;
				Instance.bSkinned = Source.bSkinned;
			}
			Gathered.Sources = MoveTemp(UniqueSources);
		}
		TArray<uint32> TriangleOffsets;
		TriangleOffsets.Reserve(Gathered.Sources.Num());
		{
			uint32 Offset = 0;
			for (const FTriangleSource& Source : Gathered.Sources)
			{
				TriangleOffsets.Add(Offset);
				Offset += Source.NumPrimitives;
			}
		}
		// Triangle entries the builder may make, as a multiple of the triangle
		// count. At 1 it may still split a triangle between leaves, as long as
		// it drops an entry elsewhere to pay for it.
		constexpr float SplitBudget = 1.0f;
		constexpr uint32 TriangleStride = GTriangleStride;

		TArray<FCachedSourceKey> Keys;
		Keys.Reserve(Gathered.Sources.Num());
		for (const FTriangleSource& Source : Gathered.Sources)
		{
			FCachedSourceKey& Key = Keys.AddDefaulted_GetRef();
			if (!Source.bSkinned)
			{
				Key.PositionSRV = Source.PositionSRV;
				Key.IndexSRV = Source.IndexSRV;
				Key.IndexBuffer = Source.IndexBuffer;
			}
			Key.FirstIndex = Source.FirstIndex;
			Key.NumPrimitives = Source.NumPrimitives;
			// A skinned source cannot be shared: its vertices are its own.
			Key.Instance = Source.bSkinned ? Source.Instance : nullptr;
		}
		for (const FTriangleSource& Source : Gathered.Sources)
		{
			GTriangleCache.AccumulatedBounds += Source.WorldBounds;
		}
		FBox AbsoluteBounds = GTriangleCache.AccumulatedBounds;
		if (!AbsoluteBounds.IsValid)
		{
			AbsoluteBounds = FBox(FVector(-1.0), FVector(1.0));
		}
		AbsoluteBounds = AbsoluteBounds.ExpandBy(AbsoluteBounds.GetSize() * 0.25);
		const FVector SnappedMin(
			FMath::FloorToDouble(AbsoluteBounds.Min.X / GFrameSnapUnits) * GFrameSnapUnits,
			FMath::FloorToDouble(AbsoluteBounds.Min.Y / GFrameSnapUnits) * GFrameSnapUnits,
			FMath::FloorToDouble(AbsoluteBounds.Min.Z / GFrameSnapUnits) * GFrameSnapUnits);
		const FVector SnappedMax(
			FMath::CeilToDouble(AbsoluteBounds.Max.X / GFrameSnapUnits) * GFrameSnapUnits,
			FMath::CeilToDouble(AbsoluteBounds.Max.Y / GFrameSnapUnits) * GFrameSnapUnits,
			FMath::CeilToDouble(AbsoluteBounds.Max.Z / GFrameSnapUnits) * GFrameSnapUnits);
		const FVector SnappedSize = SnappedMax - SnappedMin;
		const bool bBuffersValid =
			GTriangleCache.Triangles.IsValid()
			&& GTriangleCache.Nodes.IsValid();
		const bool bFrameUnchanged =
			GTriangleCache.SnappedMin.Equals(SnappedMin, 0.01)
			&& GTriangleCache.SnappedSize.Equals(SnappedSize, 0.01);
		TArray<int32> SourceSlots;
		SourceSlots.Reserve(Gathered.Sources.Num());
		bool bSupersetGrew = false;
		for (int32 SourceIndex = 0; SourceIndex < Gathered.Sources.Num(); ++SourceIndex)
		{
			if (const int32* Existing = GTriangleCache.KeyToSlot.Find(Keys[SourceIndex]))
			{
				SourceSlots.Add(*Existing);
				continue;
			}
			const int32 Slot = GTriangleCache.Keys.Num();
			GTriangleCache.KeyToSlot.Add(Keys[SourceIndex], Slot);
			GTriangleCache.Keys.Add(Keys[SourceIndex]);
			GTriangleCache.Transforms.Add(FMatrix::Identity);
			GTriangleCache.TriangleOffsets.Add(GTriangleCache.TotalTriangles);
			GTriangleCache.LocalBounds.Add(Gathered.Sources[SourceIndex].LocalBounds);
			GTriangleCache.ExactLocalBounds.Add(FBox(ForceInit));
			const uint32 SlotReserve = (uint32)FMath::Max(
				1,
				FMath::CeilToInt(
					(float)Gathered.Sources[SourceIndex].NumPrimitives * SplitBudget));
			GTriangleCache.SlotReserve.Add(SlotReserve);
			GTriangleCache.TotalTriangles += SlotReserve;
			GTriangleCache.TriangleOrders.AddDefaulted();
			GTriangleCache.TriangleTrees.AddDefaulted();
			GTriangleCache.OrderTriangleCount.Add(0);
			GTriangleCache.OrderApplied.Add(0);
			GTriangleCache.LastEnteredFrame.Add(0);
			SourceSlots.Add(Slot);
			bSupersetGrew = true;
		}
		const int32 NumSlots = GTriangleCache.Keys.Num();
		if (GTriangleCache.TriangleOrders.Num() < NumSlots)
		{
			GTriangleCache.TriangleOrders.SetNum(NumSlots);
		}
		if (GTriangleCache.TriangleTrees.Num() < NumSlots)
		{
			GTriangleCache.TriangleTrees.SetNum(NumSlots);
		}
		if (GTriangleCache.OrderTriangleCount.Num() < NumSlots)
		{
			GTriangleCache.OrderTriangleCount.SetNumZeroed(NumSlots);
		}
		if (GTriangleCache.SlotReserve.Num() < NumSlots)
		{
			GTriangleCache.SlotReserve.SetNumZeroed(NumSlots);
		}
		if (GTriangleCache.OrderApplied.Num() < NumSlots)
		{
			GTriangleCache.OrderApplied.SetNumZeroed(NumSlots);
		}
		if (GTriangleCache.TreeSerial.Num() < NumSlots)
		{
			GTriangleCache.TreeSerial.SetNumZeroed(NumSlots);
		}
		if (GTriangleCache.LayoutTrees.Num() < NumSlots)
		{
			GTriangleCache.LayoutTrees.SetNum(NumSlots);
		}
		if (GTriangleCache.LayoutRootLocal.Num() < NumSlots)
		{
			GTriangleCache.LayoutRootLocal.SetNumZeroed(NumSlots);
		}
		if (GTriangleCache.LayoutTreeKey.Num() < NumSlots)
		{
			GTriangleCache.LayoutTreeKey.SetNumZeroed(NumSlots);
		}
		if (GTriangleCache.LastEnteredFrame.Num() < NumSlots)
		{
			GTriangleCache.LastEnteredFrame.SetNumZeroed(NumSlots);
		}
		const uint64 RequiredTriangleBytes =
			(uint64)GTriangleCache.TotalTriangles * TriangleStride * sizeof(float);
		const bool bTriangleBufferFits =
			GTriangleCache.Triangles.IsValid()
			&& (uint64)GTriangleCache.Triangles->GetSize() >= RequiredTriangleBytes;
		const bool bCacheUsable = bBuffersValid && !bSupersetGrew && bTriangleBufferFits;
		const bool bReuseTriangleBuffer = bBuffersValid && bTriangleBufferFits;

		// A source's triangles survive only if the BUFFER survives. A fresh
		// allocation holds whatever the pool left in it, so every source has to
		// be written into it again however little it moved. Taking this from
		// bBuffersValid instead let a new buffer be left part filled, which
		// showed up as the extraction asserting that nothing produced it.
		const bool bTrianglesPreserved = bReuseTriangleBuffer;
		TriangleOffsets.Reset(Gathered.Sources.Num());
		for (int32 SourceIndex = 0; SourceIndex < Gathered.Sources.Num(); ++SourceIndex)
		{
			TriangleOffsets.Add(GTriangleCache.TriangleOffsets[SourceSlots[SourceIndex]]);
		}
		TArray<bool> DirtySources;
		DirtySources.Init(true, Gathered.Sources.Num());
		if (bTrianglesPreserved)
		{
			for (int32 SourceIndex = 0; SourceIndex < Gathered.Sources.Num(); ++SourceIndex)
			{
				const FTriangleSource& Source = Gathered.Sources[SourceIndex];
				const int32 Slot = SourceSlots[SourceIndex];
				DirtySources[SourceIndex] =
					Slot >= GTriangleCache.WrittenTriangles.Num()
					|| GTriangleCache.WrittenTriangles[Slot] == 0
					|| Source.bSkinned
					|| !GTriangleCache.Transforms[Slot].Equals(Source.LocalToWorld, 0.0001f);
			}
		}
		// A source nothing reached for a while gives its slot back.
		int32 NumFeedbackCulled = 0;
		{
			if (GCasterFeedbackInFlight
				&& GFrameNumberRenderThread != GCasterFeedbackFrame
				&& GCasterFeedbackReadback.IsValid()
				&& GCasterFeedbackReadback->IsReady())
			{
				const uint32* Flags =
					(const uint32*)GCasterFeedbackReadback->Lock(GCasterFeedbackCount * sizeof(uint32));
				if (Flags != nullptr)
				{
					for (uint32 Instance = 0; Instance < GCasterFeedbackCount; ++Instance)
					{
						if (Flags[Instance] == 0 || !GCasterFeedbackSlots.IsValidIndex((int32)Instance))
						{
							continue;
						}
						const int32 Slot = GCasterFeedbackSlots[(int32)Instance];
						if (GTriangleCache.LastEnteredFrame.IsValidIndex(Slot))
						{
							GTriangleCache.LastEnteredFrame[Slot] = GFrameNumberRenderThread;
						}
					}
					GCasterFeedbackReadback->Unlock();
				}
				GCasterFeedbackInFlight = false;
			}
			const uint32 Grace =
				30u;
			// Only ever skips a REWRITE. On a fresh buffer there is nothing to
			// keep, so a source nothing has reached lately still has to be
			// written or its triangles are whatever the pool left behind.
			for (int32 SourceIndex = 0;
				bReuseTriangleBuffer && SourceIndex < Gathered.Sources.Num();
				++SourceIndex)
			{
				if (!DirtySources[SourceIndex])
				{
					continue;
				}

				// A SKINNED source is never culled. Its triangles are different
				// every frame, and whether a ray entered it says nothing about
				// whether its geometry has gone stale: the placement keeps
				// moving while the mesh inside it is frozen, which reads as
				// stop motion with the shadow no longer matching its caster.
				if (Gathered.Sources[SourceIndex].bSkinned)
				{
					continue;
				}

				const int32 Slot = SourceSlots[SourceIndex];
				if (!GTriangleCache.WrittenTriangles.IsValidIndex(Slot)
					|| GTriangleCache.WrittenTriangles[Slot] == 0)
				{
					continue;
				}
				const uint32 LastEntered = GTriangleCache.LastEnteredFrame[Slot];
				const bool bRecentlyEntered =
					LastEntered != 0 && GFrameNumberRenderThread <= LastEntered + Grace;
				if (!bRecentlyEntered)
				{
					DirtySources[SourceIndex] = false;
					++NumFeedbackCulled;
				}
			}
		}
		int32 NumFramesGrown = 0;
		{
			for (int32 SourceIndex = 0; SourceIndex < Gathered.Sources.Num(); ++SourceIndex)
			{
				const int32 Slot = SourceSlots[SourceIndex];
				if (!GTriangleCache.LocalBounds.IsValidIndex(Slot))
				{
					continue;
				}
				FBox& Stored = GTriangleCache.LocalBounds[Slot];
				const FBox Grown = Stored + Gathered.Sources[SourceIndex].LocalBounds;
				if (!Grown.Min.Equals(Stored.Min, 0.0) || !Grown.Max.Equals(Stored.Max, 0.0))
				{
					Stored = Grown;
					DirtySources[SourceIndex] = true;
					++NumFramesGrown;
				}
			}
		}
		FRDGBufferRef TriangleBuffer = nullptr;
		if (GTriangleCache.TotalTriangles > 0)
		{
			if (bReuseTriangleBuffer)
			{
				TriangleBuffer = GraphBuilder.RegisterExternalBuffer(GTriangleCache.Triangles);
			}
			else
			{
				// Grown with slack so a scene that keeps gaining geometry does
				// not reallocate every frame.
				const float BufferSlack = 0.5f;
				const uint32 AllocatedTriangles = (uint32)FMath::Max(
					(int64)GTriangleCache.TotalTriangles,
					FMath::CeilToInt64((double)GTriangleCache.TotalTriangles * (1.0 + BufferSlack)));
				TriangleBuffer = GraphBuilder.CreateBuffer(
					FRDGBufferDesc::CreateBufferDesc(sizeof(float), AllocatedTriangles * TriangleStride),
					TEXT("SWRT.TraceTriangles"));
				if (bTrianglesPreserved && GTriangleCache.CachedTriangleCount > 0)
				{
					AddCopyBufferPass(
						GraphBuilder,
						TriangleBuffer, 0,
						GraphBuilder.RegisterExternalBuffer(GTriangleCache.Triangles), 0,
						(uint64)GTriangleCache.CachedTriangleCount * TriangleStride * sizeof(float));
				}
			}
		}
		TArray<uint32> SourceOrderOffsets;
		SourceOrderOffsets.Init(0xFFFFFFFFu, Gathered.Sources.Num());
		TArray<uint32> OrderUpload;
		int32 NumOrdersReady = 0;
		{
			for (int32 Index = GPendingSahBuilds.Num() - 1; Index >= 0; --Index)
			{
				FPendingSahBuild& Build = *GPendingSahBuilds[Index];
				if (Build.Task.IsValid() && !Build.Task.IsCompleted())
				{
					continue;
				}
				const int32 Slot = Build.Slot;
				if (Build.Order.Num() > 0
					&& GTriangleCache.TriangleOrders.IsValidIndex(Slot)
					&& GTriangleCache.OrderApplied.IsValidIndex(Slot)
					&& GTriangleCache.OrderTriangleCount.IsValidIndex(Slot))
				{
					GTriangleCache.TriangleOrders[Slot] = MoveTemp(Build.Order);
					if (GTriangleCache.TriangleTrees.IsValidIndex(Slot))
					{
						GTriangleCache.TriangleTrees[Slot] = MoveTemp(Build.Tree);
					}
					if (GTriangleCache.TreeSerial.IsValidIndex(Slot))
					{
						++GTriangleCache.TreeSerial[Slot];
					}
					GTriangleCache.OrderApplied[Slot] = 0;
					GTriangleCache.OrderTriangleCount[Slot] = Build.TriangleCount;
					GTriangleCache.ShapeProbe.Accumulate(Build.Probe);
					if (Build.ExactBounds.IsValid
						&& GTriangleCache.ExactLocalBounds.IsValidIndex(Slot))
					{
						GTriangleCache.ExactLocalBounds[Slot] = Build.ExactBounds;
					}
					if (Build.ExactBounds.IsValid
						&& GTriangleCache.LocalBounds.IsValidIndex(Slot))
					{
						GTriangleCache.LocalBounds[Slot] += Build.ExactBounds;
					}
				}
				GPendingSahBuilds.RemoveAtSwap(Index);
			}
			if (GOrderReadbackInFlight
				&& GFrameNumberRenderThread != GOrderReadbackFrame
				&& GOrderReadback.IsValid() && GOrderReadback->IsReady())
			{
				const uint32 FloatCount = GOrderReadbackTriangles * 9;
				const float* Positions = (const float*)GOrderReadback->Lock(FloatCount * sizeof(float));
				if (Positions != nullptr)
				{
					const bool bSahAsync =
						true;
					// What one frame is allowed to sort, in sources and in
					// triangles. The rest waits its turn.
					const int32 Budget = 4;
					const int32 TriangleBudget = 200000;
					const uint32 SahArity = GNodeArity;
					int32 Built = 0;
					int32 Copied = 0;
					for (int32 Slot = 0; Slot < GTriangleCache.Keys.Num(); ++Slot)
					{
						if (bSahAsync ? (Copied >= TriangleBudget) : (Built >= Budget))
						{
							break;
						}
						if (GTriangleCache.TriangleOrders[Slot].Num() > 0
							|| !GTriangleCache.WrittenTriangles.IsValidIndex(Slot))
						{
							continue;
						}
						if (bSahAsync && IsSahBuildPending(Slot))
						{
							continue;
						}
						const uint32 Count = GTriangleCache.WrittenTriangles[Slot];
						const uint32 Offset = GTriangleCache.TriangleOffsets[Slot];
						if (Count == 0
							|| Offset >= GOrderReadbackTriangles
							|| Count > GOrderReadbackTriangles - Offset)
						{
							continue;
						}
						FBox ExactBounds(ForceInit);
						FTriangleShapeProbe SlotProbe;
						const uint32 MaxReferences =
							GTriangleCache.SlotReserve.IsValidIndex(Slot)
								? GTriangleCache.SlotReserve[Slot]
								: Count;
						if (bSahAsync)
						{
							TUniquePtr<FPendingSahBuild> Build = MakeUnique<FPendingSahBuild>();
							Build->Slot = Slot;
							Build->TriangleCount = Count;
							Build->Triangles.SetNumUninitialized((int32)(Count * TriangleStride));
							FMemory::Memcpy(
								Build->Triangles.GetData(),
								Positions + (SIZE_T)Offset * TriangleStride,
								(SIZE_T)Count * TriangleStride * sizeof(float));
							FPendingSahBuild* Pending = Build.Get();
							Pending->Task = UE::Tasks::Launch(
								UE_SOURCE_LOCATION,
								[Pending, SahArity, MaxReferences]()
								{
									BuildTriangleOrderSAH(
										Pending->Triangles.GetData(), Pending->TriangleCount, 1,
										Pending->Order, Pending->ExactBounds, Pending->Probe,
										&Pending->Tree,
										SahArity, MaxReferences);
								});
							Copied += (int32)Count;
							GPendingSahBuilds.Add(MoveTemp(Build));
							continue;
						}
						BuildTriangleOrderSAH(
							Positions + (SIZE_T)Offset * TriangleStride, Count, 1,
							GTriangleCache.TriangleOrders[Slot], ExactBounds, SlotProbe,
							&GTriangleCache.TriangleTrees[Slot],
							SahArity, MaxReferences);
						GTriangleCache.ShapeProbe.Accumulate(SlotProbe);
						if (GTriangleCache.TriangleOrders[Slot].Num() > 0)
						{
							if (GTriangleCache.TreeSerial.IsValidIndex(Slot))
							{
								++GTriangleCache.TreeSerial[Slot];
							}
							GTriangleCache.OrderApplied[Slot] = 0;
							GTriangleCache.OrderTriangleCount[Slot] = Count;
							++Built;
							if (ExactBounds.IsValid
								&& GTriangleCache.ExactLocalBounds.IsValidIndex(Slot))
							{
								GTriangleCache.ExactLocalBounds[Slot] = ExactBounds;
							}
							if (ExactBounds.IsValid
								&& GTriangleCache.LocalBounds.IsValidIndex(Slot))
							{
								GTriangleCache.LocalBounds[Slot] += ExactBounds;
							}
						}
					}
					GOrderReadback->Unlock();
				}
				GOrderReadbackInFlight = false;
			}
			for (int32 SourceIndex = 0; SourceIndex < Gathered.Sources.Num(); ++SourceIndex)
			{
				const int32 Slot = SourceSlots[SourceIndex];
				const TArray<uint32>& Order = GTriangleCache.TriangleOrders[Slot];
				if (!GTriangleCache.OrderTriangleCount.IsValidIndex(Slot)
					|| GTriangleCache.OrderTriangleCount[Slot]
						!= Gathered.Sources[SourceIndex].NumPrimitives)
				{
					continue;
				}
				++NumOrdersReady;
				if (GTriangleCache.OrderApplied[Slot] == 0)
				{
					DirtySources[SourceIndex] = true;
				}
				if (DirtySources[SourceIndex])
				{
					SourceOrderOffsets[SourceIndex] = (uint32)OrderUpload.Num();
					OrderUpload.Append(Order);
				}
			}
		}
		TArray<uint32> SourceEntryCounts;
		SourceEntryCounts.Init(0, Gathered.Sources.Num());
		for (int32 SourceIndex = 0; SourceIndex < Gathered.Sources.Num(); ++SourceIndex)
		{
			const int32 Slot = SourceSlots[SourceIndex];
			if (!GTriangleCache.TriangleOrders.IsValidIndex(Slot)
				|| GTriangleCache.TriangleOrders[Slot].Num() == 0)
			{
				continue;
			}
			const uint32 Reserve = GTriangleCache.SlotReserve.IsValidIndex(Slot)
				? GTriangleCache.SlotReserve[Slot]
				: Gathered.Sources[SourceIndex].NumPrimitives;
			SourceEntryCounts[SourceIndex] =
				FMath::Min((uint32)GTriangleCache.TriangleOrders[Slot].Num(), Reserve);
		}
		FRDGBufferRef OrderBuffer = nullptr;
		if (OrderUpload.Num() > 0)
		{
			OrderBuffer = CreateStructuredBuffer(
				GraphBuilder, TEXT("SWRT.TraceTriangleOrder"),
				sizeof(uint32), OrderUpload.Num(), OrderUpload.GetData(),
				OrderUpload.Num() * sizeof(uint32));
		}
		TArray<uint32> WrittenTriangles;
		TriangleBuffer = ExtractTriangles(
			GraphBuilder, View, Gathered,
			TriangleBuffer, TriangleOffsets, DirtySources,
			OrderBuffer, SourceOrderOffsets, SourceEntryCounts, WrittenTriangles);
		for (int32 SourceIndex = 0; SourceIndex < Gathered.Sources.Num(); ++SourceIndex)
		{
			if (SourceOrderOffsets[SourceIndex] != 0xFFFFFFFFu)
			{
				GTriangleCache.OrderApplied[SourceSlots[SourceIndex]] = 1;
			}
		}
		if (TriangleBuffer == nullptr)
		{
			return nullptr;
		}
		{
			GTriangleCache.SnappedMin = SnappedMin;
			GTriangleCache.SnappedSize = SnappedSize;
			GTriangleCache.CachedTriangleCount = GTriangleCache.TotalTriangles;
			GTriangleCache.WrittenTriangles.SetNumZeroed(GTriangleCache.Keys.Num());
			for (int32 SourceIndex = 0; SourceIndex < Gathered.Sources.Num(); ++SourceIndex)
			{
				const int32 Slot = SourceSlots[SourceIndex];
				GTriangleCache.Transforms[Slot] = Gathered.Sources[SourceIndex].LocalToWorld;
				GTriangleCache.WrittenTriangles[Slot] = WrittenTriangles[SourceIndex];
			}
			if (!bReuseTriangleBuffer)
			{
				GraphBuilder.QueueBufferExtraction(
					TriangleBuffer, &GTriangleCache.Triangles, ERHIAccess::SRVCompute);
			}
		}
		if (bReadbackTurn && !GOrderReadbackInFlight && TriangleBuffer != nullptr
			&& GTriangleCache.TotalTriangles > 0)
		{
			bool bAnyPending = false;
			for (int32 Slot = 0; Slot < GTriangleCache.Keys.Num(); ++Slot)
			{
				if (GTriangleCache.TriangleOrders[Slot].Num() == 0
					&& GTriangleCache.WrittenTriangles.IsValidIndex(Slot)
					&& GTriangleCache.WrittenTriangles[Slot] > 0)
				{
					bAnyPending = true;
					break;
				}
			}
			if (bAnyPending)
			{
				if (!GOrderReadback.IsValid())
				{
					GOrderReadback = MakeUnique<FRHIGPUBufferReadback>(
						TEXT("SWRT.TraceTriangleOrder"));
				}
				const uint32 BufferTriangles =
					TriangleBuffer->GetSize() / (uint32)(TriangleStride * sizeof(float));
				GOrderReadbackTriangles =
					FMath::Min(GTriangleCache.TotalTriangles, BufferTriangles);
				if (GOrderReadbackTriangles > 0)
				{
					AddEnqueueCopyPass(
						GraphBuilder, GOrderReadback.Get(), TriangleBuffer,
						(uint32)GOrderReadbackTriangles * TriangleStride * sizeof(float));
					GOrderReadbackInFlight = true;
					GOrderReadbackFrame = GFrameNumberRenderThread;
				}
			}
		}
		Timing.Slots = FPlatformTime::Seconds() - StageStart;
		StageStart = FPlatformTime::Seconds();
		const FVector Translation = View.ViewMatrices.GetPreViewTranslation();
		FBox SceneBounds(ForceInit);
		for (int32 SourceIndex = 0; SourceIndex < Gathered.Sources.Num(); ++SourceIndex)
		{
			if (WrittenTriangles[SourceIndex] > 0)
			{
				SceneBounds += Gathered.Sources[SourceIndex].WorldBounds;
			}
		}
		if (!SceneBounds.IsValid)
		{
			SceneBounds = FBox(FVector(-1.0), FVector(1.0));
		}
		SceneBounds = SceneBounds.ExpandBy(SceneBounds.GetSize() * 0.25);
		const FVector FrameMin = SnappedMin;
		const FVector FrameSize = SnappedSize;
		const FVector3f BuildSceneMin = (FVector3f)FrameMin;
		const FVector3f SceneMin = (FVector3f)(FrameMin + Translation);
		const FVector3f QuantScale(
			(float)(65534.0 / FMath::Max(FrameSize.X, 1.0)),
			(float)(65534.0 / FMath::Max(FrameSize.Y, 1.0)),
			(float)(65534.0 / FMath::Max(FrameSize.Z, 1.0)));
		TArray<FVector4f> SourceFrames;
		TArray<FVector4f> InstanceData;
		TArray<FVector4f> InstanceTransforms;
		struct FPlacementRecord
		{
			FBox WorldBounds = FBox(ForceInit);
			FVector4f Rows[3];
			FVector3f FrameMin = FVector3f::ZeroVector;
			FVector3f FrameScale = FVector3f::OneVector;
			int32 Slot = INDEX_NONE;
		};
		TArray<FPlacementRecord> Placements;
		TArray<FBox> PlacementBounds;
		double BoundsSlackSum = 0.0;
		double BoundsSlackMax = 0.0;
		int32 BoundsSlackCount = 0;
		FInstanceTree InstanceTree;
		{
			SourceFrames.Reserve(GTriangleCache.LocalBounds.Num() * 2);
			for (const FBox& SlotBounds : GTriangleCache.LocalBounds)
			{
				const FVector FrameLocalSize = SlotBounds.GetSize();
				SourceFrames.Add(FVector4f((FVector3f)SlotBounds.Min, 0.0f));
				SourceFrames.Add(FVector4f(FVector3f(
					(float)(65534.0 / FMath::Max(FrameLocalSize.X, 1.0)),
					(float)(65534.0 / FMath::Max(FrameLocalSize.Y, 1.0)),
					(float)(65534.0 / FMath::Max(FrameLocalSize.Z, 1.0))), 0.0f));
			}
			Placements.Reserve(Gathered.Instances.Num());
			PlacementBounds.Reserve(Gathered.Instances.Num());
			for (const FGatheredInstance& Placement : Gathered.Instances)
			{
				const int32 PlacementSlot = SourceSlots[Placement.SourceIndex];
				const FBox& SlotLocalBounds = GTriangleCache.LocalBounds[PlacementSlot];
				const FVector LocalSize = SlotLocalBounds.GetSize();
				FPlacementRecord& Record = Placements.AddDefaulted_GetRef();
				Record.Slot = PlacementSlot;
				Record.FrameMin = (FVector3f)SlotLocalBounds.Min;
				Record.FrameScale = FVector3f(
					(float)(65534.0 / FMath::Max(LocalSize.X, 1.0)),
					(float)(65534.0 / FMath::Max(LocalSize.Y, 1.0)),
					(float)(65534.0 / FMath::Max(LocalSize.Z, 1.0)));
				const FMatrix LocalToTranslatedWorld =
					Placement.LocalToWorld * FTranslationMatrix(Translation);
				for (int32 Corner = 0; Corner < 8; ++Corner)
				{
					const FVector LocalCorner(
						(Corner & 1) ? SlotLocalBounds.Max.X : SlotLocalBounds.Min.X,
						(Corner & 2) ? SlotLocalBounds.Max.Y : SlotLocalBounds.Min.Y,
						(Corner & 4) ? SlotLocalBounds.Max.Z : SlotLocalBounds.Min.Z);
					Record.WorldBounds += LocalToTranslatedWorld.TransformPosition(LocalCorner);
				}
				const FMatrix44f WorldToLocal = FMatrix44f(LocalToTranslatedWorld.Inverse());
				for (int32 Row = 0; Row < 3; ++Row)
				{
					Record.Rows[Row] = FVector4f(
						WorldToLocal.M[0][Row], WorldToLocal.M[1][Row],
						WorldToLocal.M[2][Row], WorldToLocal.M[3][Row]);
				}
				if (GTriangleCache.ExactLocalBounds.IsValidIndex(PlacementSlot)
					&& GTriangleCache.ExactLocalBounds[PlacementSlot].IsValid)
				{
					const FBox& Exact = GTriangleCache.ExactLocalBounds[PlacementSlot];
					FBox TightWorld(ForceInit);
					for (int32 Corner = 0; Corner < 8; ++Corner)
					{
						const FVector LocalCorner(
							(Corner & 1) ? Exact.Max.X : Exact.Min.X,
							(Corner & 2) ? Exact.Max.Y : Exact.Min.Y,
							(Corner & 4) ? Exact.Max.Z : Exact.Min.Z);
						TightWorld += LocalToTranslatedWorld.TransformPosition(LocalCorner);
					}
					auto HalfArea = [](const FBox& Box) -> double
					{
						const FVector S = Box.GetSize();
						return S.X * S.Y + S.Y * S.Z + S.Z * S.X;
					};
					const double TightArea = HalfArea(TightWorld);
					if (TightArea > UE_SMALL_NUMBER)
					{
						const double Ratio = HalfArea(Record.WorldBounds) / TightArea;
						BoundsSlackSum += Ratio;
						BoundsSlackMax = FMath::Max(BoundsSlackMax, Ratio);
						++BoundsSlackCount;
					}
				}
				PlacementBounds.Add(Record.WorldBounds);
			}
		}
		FNodeLayout& Layout = GTriangleCache.Layout;
		FRDGBufferRef NodeBuffer = nullptr;
		{
			constexpr uint32 LeafSize = (uint32)GLeafTriangles;
			constexpr uint32 Arity = GNodeArity;
			int32 FreeTreeCount = 0;
			{
				for (const TArray<FSahBuildNode>& Tree : GTriangleCache.TriangleTrees)
				{
					FreeTreeCount += Tree.Num() > 0 ? 1 : 0;
				}
			}
			// Collapse a subtree into one leaf once it holds at most this many
			// triangles. A node holding two or three children costs more to
			// visit than its triangles cost to test.
			constexpr uint32 CollapseBudgetNow = 6;
			const bool bLayoutAsync =
				true;
			StageStart = FPlatformTime::Seconds();
			bool bNodeLayoutChanged = false;
			if (GPendingLayout.IsValid()
				&& (!GPendingLayout->Task.IsValid() || GPendingLayout->Task.IsCompleted()))
			{
				Layout = MoveTemp(GPendingLayout->Out);
				GTriangleCache.LayoutFreeTreeCount = GPendingLayout->FreeTreeCount;
				GTriangleCache.LayoutCollapseBudget = GPendingLayout->CollapseBudget;
				GTriangleCache.LeafSize = GPendingLayout->LeafSize;
				GTriangleCache.Arity = GPendingLayout->Arity;
				GPendingLayout.Reset();
				bNodeLayoutChanged = true;
			}
			const bool bLayoutStale =
				!bCacheUsable
				|| Layout.Descs.Num() == 0
				|| GTriangleCache.LeafSize != LeafSize
				|| GTriangleCache.Arity != Arity
				|| GTriangleCache.LayoutFreeTreeCount != FreeTreeCount
				|| GTriangleCache.LayoutCollapseBudget != CollapseBudgetNow;
			const bool bNoLayoutYet = Layout.Descs.Num() == 0;
			bool bVisibleSourceMissingRoot = false;
			for (int32 SourceIndex = 0; SourceIndex < Gathered.Sources.Num(); ++SourceIndex)
			{
				const int32 Slot = SourceSlots[SourceIndex];
				if (!GTriangleCache.WrittenTriangles.IsValidIndex(Slot)
					|| GTriangleCache.WrittenTriangles[Slot] == 0)
				{
					continue;
				}
				if (!Layout.Roots.IsValidIndex(Slot) || Layout.Roots[Slot] == INDEX_NONE)
				{
					bVisibleSourceMissingRoot = true;
					break;
				}
			}
			if (bLayoutStale && (!bLayoutAsync || bNoLayoutYet || bVisibleSourceMissingRoot))
			{
				DrainLayoutBuild();
				Layout = FNodeLayout();
				BuildNodeLayout(
					GTriangleCache.TriangleOffsets, GTriangleCache.WrittenTriangles, LeafSize, Arity,
					TConstArrayView<TArray<FSahBuildNode>>(GTriangleCache.TriangleTrees),
					CollapseBudgetNow,
					false,
					&GTriangleCache.LayoutTrees, &GTriangleCache.LayoutRootLocal,
					&GTriangleCache.LayoutTreeKey, GTriangleCache.TreeSerial,
					Layout);
				GTriangleCache.LayoutFreeTreeCount = FreeTreeCount;
				GTriangleCache.LayoutCollapseBudget = CollapseBudgetNow;
				GTriangleCache.LeafSize = LeafSize;
				GTriangleCache.Arity = Arity;
				bNodeLayoutChanged = true;
			}
			else if (bLayoutStale && !GPendingLayout.IsValid())
			{
				TUniquePtr<FPendingLayoutBuild> Build = MakeUnique<FPendingLayoutBuild>();
				Build->TriangleOffsets = GTriangleCache.TriangleOffsets;
				Build->TriangleCounts = GTriangleCache.WrittenTriangles;
				Build->FreeTrees = GTriangleCache.TriangleTrees;
				Build->LeafSize = LeafSize;
				Build->Arity = Arity;
				Build->CollapseBudget = CollapseBudgetNow;
				Build->FreeTreeCount = FreeTreeCount;

				FPendingLayoutBuild* Pending = Build.Get();
				Pending->Task = UE::Tasks::Launch(
					UE_SOURCE_LOCATION,
					[Pending]()
					{
						BuildNodeLayout(
							Pending->TriangleOffsets, Pending->TriangleCounts,
							Pending->LeafSize, Pending->Arity,
							Pending->FreeTrees,
							Pending->CollapseBudget, Pending->bTreeStats,
							nullptr, nullptr, nullptr, TConstArrayView<uint32>(),
							Pending->Out);
					});
				GPendingLayout = MoveTemp(Build);
			}
			const bool bReuseLayout = !bNodeLayoutChanged;
			Timing.bRebuiltLayout = bNodeLayoutChanged;
			Timing.Layout = FPlatformTime::Seconds() - StageStart;
			StageStart = FPlatformTime::Seconds();
			TArray<bool> DirtySlots;
			DirtySlots.Init(false, GTriangleCache.Keys.Num());
			for (int32 SourceIndex = 0; SourceIndex < Gathered.Sources.Num(); ++SourceIndex)
			{
				if (DirtySources[SourceIndex] && DirtySlots.IsValidIndex(SourceSlots[SourceIndex]))
				{
					DirtySlots[SourceSlots[SourceIndex]] = true;
				}
			}
			TArray<uint32> FilteredNodeIndices;
			TArray<FIntPoint> FilteredLevels;
			TConstArrayView<uint32> RefitNodeIndices = Layout.LevelNodeIndices;
			TConstArrayView<FIntPoint> RefitLevels = Layout.Levels;
			const bool bRefitEverything = !bReuseLayout || Layout.SourceLevelSlices.Num() == 0;
			if (!bRefitEverything)
			{
				FilteredNodeIndices.Reserve(Layout.LevelNodeIndices.Num() / 4);
				FilteredLevels.Reserve(Layout.Levels.Num());
				for (int32 Level = 0; Level < Layout.Levels.Num(); ++Level)
				{
					const int32 LevelStart = FilteredNodeIndices.Num();
					if (Level >= Layout.NumBlasLevels)
					{
						const FIntPoint Whole = Layout.Levels[Level];
						for (int32 Node = 0; Node < Whole.Y; ++Node)
						{
							FilteredNodeIndices.Add(Layout.LevelNodeIndices[Whole.X + Node]);
						}
						FilteredLevels.Add(FIntPoint(LevelStart, FilteredNodeIndices.Num() - LevelStart));
						continue;
					}
					for (int32 SourceIndex = 0; SourceIndex < DirtySlots.Num(); ++SourceIndex)
					{
						if (!DirtySlots[SourceIndex]
							|| !Layout.SourceLevelSlices.IsValidIndex(SourceIndex)
							|| !Layout.SourceLevelSlices[SourceIndex].IsValidIndex(Level))
						{
							continue;
						}
						const FIntPoint Slice = Layout.SourceLevelSlices[SourceIndex][Level];
						for (int32 Node = 0; Node < Slice.Y; ++Node)
						{
							FilteredNodeIndices.Add(Layout.LevelNodeIndices[Slice.X + Node]);
						}
					}
					FilteredLevels.Add(FIntPoint(LevelStart, FilteredNodeIndices.Num() - LevelStart));
				}
				RefitNodeIndices = FilteredNodeIndices;
				RefitLevels = FilteredLevels;
			}
			Timing.RefitList = FPlatformTime::Seconds() - StageStart;
			StageStart = FPlatformTime::Seconds();
			TArray<bool> SlotVisible;
			SlotVisible.Init(false, GTriangleCache.Keys.Num());
			for (int32 SourceIndex = 0; SourceIndex < Gathered.Sources.Num(); ++SourceIndex)
			{
				if (SlotVisible.IsValidIndex(SourceSlots[SourceIndex]))
				{
					SlotVisible[SourceSlots[SourceIndex]] = true;
				}
			}
			TArray<uint32> InstanceVisible;
			InstanceVisible.Reserve(Layout.InstanceSlots.Num());
			for (int32 Slot : Layout.InstanceSlots)
			{
				InstanceVisible.Add(SlotVisible.IsValidIndex(Slot) && SlotVisible[Slot] ? 1u : 0u);
			}
			NodeBuffer = BuildNodeBounds(
				GraphBuilder, View, TriangleBuffer, Layout, BuildSceneMin, QuantScale, SourceFrames,
				InstanceVisible, Layout.NumBlasLevels,
				RefitNodeIndices, RefitLevels, &GTriangleCache, bReuseLayout);
		}
		if (NodeBuffer == nullptr)
		{
			const FUintVector4 Empty[2] = { FUintVector4(0, 0, 0, 0), FUintVector4(0, 0, 0, 0) };
			NodeBuffer = CreateStructuredBuffer(
				GraphBuilder, TEXT("SWRT.TraceNodesEmpty"),
				sizeof(FUintVector4), 2, Empty, sizeof(Empty));
		}
		Timing.Instances = FPlatformTime::Seconds() - StageStart;
		TArray<FVector4f> CasterData;
		CasterData.Reserve(Gathered.Sources.Num() * 2);
		for (int32 SourceIndex = 0; SourceIndex < Gathered.Sources.Num(); ++SourceIndex)
		{
			const FTriangleSource& Source = Gathered.Sources[SourceIndex];
			const int32 Root = Layout.Roots.IsValidIndex(SourceIndex) ? Layout.Roots[SourceIndex] : INDEX_NONE;
			const bool bEmpty = WrittenTriangles[SourceIndex] == 0 || Root == INDEX_NONE;
			const FVector Min = bEmpty ? FVector(UE_BIG_NUMBER) : Source.WorldBounds.Min;
			const FVector Max = bEmpty ? FVector(-UE_BIG_NUMBER) : Source.WorldBounds.Max;
			const FVector Offset = bEmpty ? FVector::ZeroVector : Translation;
			const uint32 Entry = (uint32)FMath::Max(Root, 0);
			CasterData.Add(FVector4f(
				(float)(Min.X + Offset.X), (float)(Min.Y + Offset.Y), (float)(Min.Z + Offset.Z),
				(float)Entry));
			CasterData.Add(FVector4f(
				(float)(Max.X + Offset.X), (float)(Max.Y + Offset.Y), (float)(Max.Z + Offset.Z),
				(float)WrittenTriangles[SourceIndex]));
		}
		FRDGBufferRef CasterBuffer = CreateStructuredBuffer(
			GraphBuilder, TEXT("SWRT.TraceCasters"),
			sizeof(FVector4f), CasterData.Num(), CasterData.GetData(),
			CasterData.Num() * sizeof(FVector4f));
		TArray<uint32> RootIndices;
		RootIndices.Reserve(Gathered.Sources.Num());
		{
			for (int32 SourceIndex = 0; SourceIndex < Gathered.Sources.Num(); ++SourceIndex)
			{
				if (Layout.Roots.IsValidIndex(SourceIndex) && Layout.Roots[SourceIndex] != INDEX_NONE)
				{
					RootIndices.Add((uint32)Layout.Roots[SourceIndex]);
				}
			}
		}
		const uint32 NumRoots = (uint32)RootIndices.Num();
		if (RootIndices.Num() == 0)
		{
			RootIndices.Add(0);
		}
		FRDGBufferRef RootBuffer = CreateStructuredBuffer(
			GraphBuilder, TEXT("SWRT.TraceRoots"),
			sizeof(uint32), RootIndices.Num(), RootIndices.GetData(),
			RootIndices.Num() * sizeof(uint32));
		TArray<int32> InstanceSlots;
		uint32 NumInstances = 0;
		if (Placements.Num() > 0)
		{
			constexpr uint32 InstanceArity = GNodeArity;
			BuildInstanceTree(
				PlacementBounds, InstanceArity, SceneMin, QuantScale,
				Light.TranslatedPosition, InstanceTree);
			InstanceData.Reserve(Placements.Num() * 2);
			InstanceTransforms.Reserve(Placements.Num() * 5);
			InstanceSlots.Reserve(InstanceTree.Order.Num());
			for (int32 TreeSlot = 0; TreeSlot < InstanceTree.Order.Num(); ++TreeSlot)
			{
				const FPlacementRecord& Record = Placements[InstanceTree.Order[TreeSlot]];
				const int32 Root =
					Layout.Roots.IsValidIndex(Record.Slot) ? Layout.Roots[Record.Slot] : INDEX_NONE;
				InstanceSlots.Add(Record.Slot);
				InstanceData.Add(FVector4f((FVector3f)Record.WorldBounds.Min, (float)FMath::Max(Root, 0)));
				InstanceData.Add(FVector4f((FVector3f)Record.WorldBounds.Max, Root == INDEX_NONE ? -1.0f : 1.0f));
				InstanceTransforms.Add(Record.Rows[0]);
				InstanceTransforms.Add(Record.Rows[1]);
				InstanceTransforms.Add(Record.Rows[2]);
				InstanceTransforms.Add(FVector4f(Record.FrameMin, 0.0f));
				InstanceTransforms.Add(FVector4f(Record.FrameScale, 0.0f));
			}
			NumInstances = (uint32)InstanceTree.Order.Num();
		}

		if (NumInstances == 0)
		{
			return nullptr;
		}
		if (InstanceData.Num() == 0)
		{
			InstanceData.Add(FVector4f(ForceInitToZero));
			InstanceData.Add(FVector4f(ForceInitToZero));
		}
		if (InstanceTransforms.Num() == 0)
		{
			InstanceTransforms.AddZeroed(5);
		}
		if (InstanceTree.Nodes.Num() == 0)
		{
			InstanceTree.Nodes.AddZeroed(1);
			InstanceTree.Root = 0;
		}
		FRDGBufferRef InstanceBuffer = CreateStructuredBuffer(
			GraphBuilder, TEXT("SWRT.TraceInstances"),
			sizeof(FVector4f), InstanceData.Num(), InstanceData.GetData(),
			InstanceData.Num() * sizeof(FVector4f));
		FRDGBufferRef InstanceTransformBuffer = CreateStructuredBuffer(
			GraphBuilder, TEXT("SWRT.TraceInstanceTransforms"),
			sizeof(FVector4f), InstanceTransforms.Num(), InstanceTransforms.GetData(),
			InstanceTransforms.Num() * sizeof(FVector4f));
		FRDGBufferRef InstanceNodeBuffer = CreateStructuredBuffer(
			GraphBuilder, TEXT("SWRT.TraceInstanceNodes"),
			sizeof(FUintVector4), InstanceTree.Nodes.Num(), InstanceTree.Nodes.GetData(),
			InstanceTree.Nodes.Num() * sizeof(FUintVector4));
		if (TriangleBuffer != nullptr)
		{
			GraphBuilder.QueueBufferExtraction(
				TriangleBuffer, &GTraceTriangles, ERHIAccess::SRVCompute);
		}
		static bool bLogged = false;
		if (!bLogged)
		{
			bLogged = true;
			const bool bIndexSRVsAvailable = AreBufferSRVsAlwaysCreatedByDefault(GMaxRHIShaderPlatform);
			UE_LOG(LogTemp, Log,
				TEXT("SWRT.Trace: %d casters -> %d sources, %u triangles ")
				TEXT("(%d static, %d skinned, %d skipped). Index SRVs available by default: %s"),
				Casters.Num(), Gathered.Sources.Num(), Gathered.TotalTriangles,
				Gathered.NumStaticCasters, Gathered.NumSkinnedCasters, Gathered.NumSkippedCasters,
				bIndexSRVsAvailable ? TEXT("yes") : TEXT("NO - indices would need copying"));
		}
		const FRDGTextureDesc MaskDesc = FRDGTextureDesc::Create2D(
			SceneDepthTexture->Desc.Extent,
			PF_G8,
			FClearValueBinding::White,
			TexCreate_ShaderResource | TexCreate_UAV);
		FRDGTextureRef MaskTexture = GraphBuilder.CreateTexture(MaskDesc, TEXT("SWRT.TraceMask"));
		FIntRect TraceRect = View.ViewRect;
		{
			constexpr double GConeRectMaxAreaFraction = 0.75;
			FIntRect ConeRect;
			if (ComputeConeScreenRect(View, Light, ConeRect)
				&& (double)ConeRect.Area()
					< (double)View.ViewRect.Area() * GConeRectMaxAreaFraction)
			{
				// The rect shrank, so the pixels outside it were never written.
				// Clearing to lit is exact rather than a fallback: a pixel
				// outside the cone's screen rect cannot be inside the cone.
				AddClearUAVPass(
					GraphBuilder, GraphBuilder.CreateUAV(MaskTexture),
					FVector4f(1.0f, 1.0f, 1.0f, 1.0f));
				TraceRect = ConeRect;
			}
		}
		FSWRTTraceCS::FParameters* Parameters =
			GraphBuilder.AllocParameters<FSWRTTraceCS::FParameters>();
		Parameters->View = View.ViewUniformBuffer;
		Parameters->SceneTextures = CreateSceneTextureShaderParameters(
			GraphBuilder, View.GetSceneTexturesChecked(), View.FeatureLevel, ESceneTextureSetupMode::All);
		Parameters->SWRTTriangles = GraphBuilder.CreateSRV(
			TriangleBuffer, PF_A32B32G32R32F);
		Parameters->SWRTCasters = GraphBuilder.CreateSRV(CasterBuffer, PF_A32B32G32R32F);
		Parameters->SWRTRoots = GraphBuilder.CreateSRV(RootBuffer, PF_R32_UINT);
		Parameters->SWRTInstances = GraphBuilder.CreateSRV(InstanceBuffer, PF_A32B32G32R32F);
		Parameters->SWRTInstanceTransforms = GraphBuilder.CreateSRV(InstanceTransformBuffer, PF_A32B32G32R32F);
		Parameters->SWRTInstanceNodes = GraphBuilder.CreateSRV(InstanceNodeBuffer, PF_R32G32B32A32_UINT);
		Parameters->SWRTInstanceRoot = (uint32)FMath::Max(InstanceTree.Root, 0);
		Parameters->SWRTNodes = GraphBuilder.CreateSRV(NodeBuffer, PF_R32G32B32A32_UINT);
		Parameters->SWRTSceneMin = SceneMin;
		Parameters->SWRTQuantScale = QuantScale;
		// Triangles are in absolute world space, the view is in translated world
		// space, and this is the difference.
		Parameters->SWRTTriangleSpaceOffset = (FVector3f)(-Translation);
		Parameters->SWRTCasterCount = NumInstances;
		const bool bStats = SWRTSpot::GetDebugMode() != 0;
		const uint32 ZeroStats[64] = {};
		FRDGBufferRef StatsBuffer = CreateStructuredBuffer(
			GraphBuilder, TEXT("SWRT.TraceStats"),
			sizeof(uint32), UE_ARRAY_COUNT(ZeroStats), ZeroStats, sizeof(ZeroStats));
		Parameters->SWRTStats = GraphBuilder.CreateUAV(StatsBuffer, PF_R32_UINT);
		const uint32 FeedbackSlots = FMath::Max(NumInstances, 1u);
		FRDGBufferRef CasterHitsBuffer = GraphBuilder.CreateBuffer(
			FRDGBufferDesc::CreateBufferDesc(sizeof(uint32), FeedbackSlots),
			TEXT("SWRT.TraceCasterHits"));
		FRDGBufferUAVRef CasterHitsUAV = GraphBuilder.CreateUAV(CasterHitsBuffer, PF_R32_UINT);
		AddClearUAVPass(GraphBuilder, CasterHitsUAV, 0u);
		Parameters->SWRTCasterHitsOut = CasterHitsUAV;
		Parameters->SWRTSceneDepth = SceneDepthTexture;
		Parameters->SWRTOutMask = GraphBuilder.CreateUAV(MaskTexture);
		Parameters->SWRTLightPosition = (FVector3f)Light.TranslatedPosition;
		Parameters->SWRTLightRadius = Light.Radius;
		Parameters->SWRTLightDirection = (FVector3f)Light.Direction;
		Parameters->SWRTCosOuterCone = Light.CosOuterCone;
		Parameters->SWRTPixelMin = FUintVector2((uint32)TraceRect.Min.X, (uint32)TraceRect.Min.Y);
		Parameters->SWRTPixelMax = FUintVector2((uint32)TraceRect.Max.X, (uint32)TraceRect.Max.Y);
		Parameters->SWRTRayStartBias = Light.RayStartBias;
		Parameters->SWRTRayStartSlope = Light.RayStartSlope;
		FSWRTTraceCS::FPermutationDomain PermutationVector;
		PermutationVector.Set<FSWRTTraceCS::FStatsDim>(bStats);

		TShaderMapRef<FSWRTTraceCS> ComputeShader(
			GetGlobalShaderMap(View.GetFeatureLevel()), PermutationVector);
		const FIntPoint TracedPixels = TraceRect.Size();

		// 8x16, so a warp still covers the same 8x4 pixels and the group is two
		// vertically adjacent TILES rather than one wider one. 16x8 would give a
		// warp a 16x2 strip and widen what a wave covers.
		const FUintVector2 GroupSize = FSWRTTraceCS::GetGroupSize();
		const FUintVector2 DispatchGrid(
			FMath::DivideAndRoundUp((uint32)TracedPixels.X, GroupSize.X),
			FMath::DivideAndRoundUp((uint32)TracedPixels.Y, GroupSize.Y));
		const FUintVector2 TileGrid(
			FMath::DivideAndRoundUp((uint32)TracedPixels.X, 8u),
			FMath::DivideAndRoundUp((uint32)TracedPixels.Y, 8u));
		{
			const float HeatmapRange =
				(float)FMath::Max(CVarTraceHeatmapRange.GetValueOnRenderThread(), 1);

			// Only the stats variant has the counters these views read.
			const uint32 DebugView = bStats ? SWRTSpot::GetTraceView() : 0u;

			Parameters->SWRTDebugView = DebugView;
			Parameters->SWRTHeatmapScale = DebugView != 0u ? (1.0f / HeatmapRange) : 0.0f;
		}

		Parameters->SWRTDispatchGrid = DispatchGrid;
		Parameters->SWRTTileGrid = TileGrid;
		{
			constexpr uint32 TileStride = GTileListStride;
			const uint32 TileCount = TileGrid.X * TileGrid.Y;
			FRDGBufferRef TileInstanceBuffer = nullptr;
			if (TileCount > 0)
			{
				FRDGBufferRef TileBoundsBuffer = GraphBuilder.CreateBuffer(
					FRDGBufferDesc::CreateBufferDesc(sizeof(FVector4f), TileCount * 2),
					TEXT("SWRT.TraceTileBounds"));
				{
					FSWRTTileBoundsCS::FParameters* BoundsParameters =
						GraphBuilder.AllocParameters<FSWRTTileBoundsCS::FParameters>();
					BoundsParameters->View = View.ViewUniformBuffer;
					BoundsParameters->SWRTTbSceneDepth = SceneDepthTexture;
					BoundsParameters->SWRTTbOutBounds =
						GraphBuilder.CreateUAV(TileBoundsBuffer, PF_A32B32G32R32F);
					BoundsParameters->SWRTTbPixelMin =
						FUintVector2((uint32)TraceRect.Min.X, (uint32)TraceRect.Min.Y);
					BoundsParameters->SWRTTbPixelMax =
						FUintVector2((uint32)TraceRect.Max.X, (uint32)TraceRect.Max.Y);
					BoundsParameters->SWRTTbTileGrid = TileGrid;
					BoundsParameters->SWRTTbLightPosition = (FVector3f)Light.TranslatedPosition;
					BoundsParameters->SWRTTbLightRadius = Light.Radius;
					BoundsParameters->SWRTTbLightDirection = (FVector3f)Light.Direction;
					BoundsParameters->SWRTTbCosOuterCone = Light.CosOuterCone;
					// A rect that is a whole number of tiles needs no bounds check
					// on the depth load or on the write.
					const bool bExactGrid =
						(TracedPixels.X % 8) == 0 && (TracedPixels.Y % 8) == 0;

					FSWRTTileBoundsCS::FPermutationDomain BoundsPermutation;
					BoundsPermutation.Set<FSWRTTileBoundsCS::FExactGridDim>(bExactGrid);

					TShaderMapRef<FSWRTTileBoundsCS> BoundsShader(
						GetGlobalShaderMap(View.GetFeatureLevel()), BoundsPermutation);

					FComputeShaderUtils::AddPass(
						GraphBuilder,
						RDG_EVENT_NAME("SWRT.TraceTileBounds %ux%u tiles",
							TileGrid.X, TileGrid.Y),
						BoundsShader,
						BoundsParameters,
						FIntVector(TileGrid.X, TileGrid.Y, 1));
				}
				TileInstanceBuffer = GraphBuilder.CreateBuffer(
					FRDGBufferDesc::CreateBufferDesc(sizeof(uint32), TileCount * TileStride),
					TEXT("SWRT.TraceTileInstances"));
				{
					FSWRTTileInstanceCS::FParameters* ListParameters =
						GraphBuilder.AllocParameters<FSWRTTileInstanceCS::FParameters>();
					ListParameters->SWRTTiNodes =
						GraphBuilder.CreateSRV(InstanceNodeBuffer, PF_R32G32B32A32_UINT);
					ListParameters->SWRTTiInstanceRoot = (uint32)FMath::Max(InstanceTree.Root, 0);
					ListParameters->SWRTTiCasterCount = NumInstances;
					ListParameters->SWRTTiBounds =
						GraphBuilder.CreateSRV(TileBoundsBuffer, PF_A32B32G32R32F);
					ListParameters->SWRTTiOutInstances =
						GraphBuilder.CreateUAV(TileInstanceBuffer, PF_R32_UINT);
					ListParameters->SWRTTiStride = TileStride;
					ListParameters->SWRTTiTileGrid = TileGrid;
					ListParameters->SWRTTiSceneMin = SceneMin;
					ListParameters->SWRTTiQuantScale = QuantScale;
					ListParameters->SWRTTiLightPosition = (FVector3f)Light.TranslatedPosition;
					TShaderMapRef<FSWRTTileInstanceCS> ListShader(
						GetGlobalShaderMap(View.GetFeatureLevel()));

					constexpr uint32 ListGroupSize = FSWRTTileInstanceCS::ThreadGroupSize;
					FComputeShaderUtils::AddPass(
						GraphBuilder,
						RDG_EVENT_NAME("SWRT.TraceTileInstances %u tiles, %u instances, %u per group",
							TileCount, NumInstances, ListGroupSize),
						ListShader,
						ListParameters,
						FIntVector(
							FMath::DivideAndRoundUp(TileCount, ListGroupSize), 1, 1));
				}
			}
			if (TileInstanceBuffer == nullptr)
			{
				const uint32 Empty = 0;
				TileInstanceBuffer = CreateStructuredBuffer(
					GraphBuilder, TEXT("SWRT.TraceTileInstancesDummy"),
					sizeof(uint32), 1, &Empty, sizeof(uint32));
			}
			Parameters->SWRTTileInstances =
				GraphBuilder.CreateSRV(TileInstanceBuffer, PF_R32_UINT);
			Parameters->SWRTTileInstanceStride = TileStride;
		}
		// Groups per row in the dispatch swizzle, which keeps groups that run
		// together on the same part of the screen and reading the same part of
		// the BVH.
		constexpr int32 TileWidthSetting = 4;
		Parameters->SWRTSwizzleTileWidth = TileWidthSetting <= 0
			? FMath::Max(DispatchGrid.X, 1u)
			: (uint32)FMath::Clamp(TileWidthSetting, 1, 64);
		{
			const FSWRTTraceCS::FSwizzleConstants Swizzle =
				FSWRTTraceCS::MakeSwizzleConstants(
					DispatchGrid, Parameters->SWRTSwizzleTileWidth);
			Parameters->SWRTSwizzlePerfectTileCTAs = Swizzle.PerfectTileCTAs;
			Parameters->SWRTSwizzleTotalPerfectCTAs = Swizzle.TotalPerfectCTAs;
			Parameters->SWRTSwizzleLastTileWidth = Swizzle.LastTileWidth;
			Parameters->SWRTSwizzleMagic = Swizzle.Magic;
			Parameters->SWRTSwizzleMagicShift = Swizzle.MagicShift;
		}
		FComputeShaderUtils::AddPass(
			GraphBuilder,
			RDG_EVENT_NAME("SWRT.Trace %d sources, %u tris, %d nodes",
				Gathered.Sources.Num(), Gathered.TotalTriangles, Layout.Descs.Num()),
			ComputeShader,
			Parameters,
			FIntVector(DispatchGrid.X, DispatchGrid.Y, 1));
		if (bReadbackTurn && NumInstances > 0 && !GCasterFeedbackInFlight)
		{
			if (!GCasterFeedbackReadback.IsValid())
			{
				GCasterFeedbackReadback = MakeUnique<FRHIGPUBufferReadback>(
					TEXT("SWRT.TraceCasterHits"));
			}
			GCasterFeedbackSlots = InstanceSlots;
			GCasterFeedbackCount = NumInstances;
			AddEnqueueCopyPass(
				GraphBuilder, GCasterFeedbackReadback.Get(), CasterHitsBuffer,
				NumInstances * sizeof(uint32));
			GCasterFeedbackInFlight = true;
			GCasterFeedbackFrame = GFrameNumberRenderThread;
		}
		if (bStats)
		{
			if (!GStatsReadback.IsValid())
			{
				GStatsReadback = MakeUnique<FRHIGPUBufferReadback>(TEXT("SWRT.TraceStats"));
			}
		Timing.Total = FPlatformTime::Seconds() - TickStart;
			if (GStatsReadbackInFlight
				&& GFrameNumberRenderThread != GStatsReadbackFrame
				&& GStatsReadback->IsReady())
			{
				GStatsReadbackInFlight = false;
				const uint32* Counts = (const uint32*)GStatsReadback->Lock(64 * sizeof(uint32));
				if (Counts != nullptr)
				{
					const double Rays = (double)FMath::Max(Counts[0], 1u);
					TArray<FString> LogLines;
					auto Emit = [&LogLines](const FString& Line)
					{
						UE_LOG(LogTemp, Log, TEXT("%s"), *Line);
						LogLines.Add(Line);
					};
					ON_SCOPE_EXIT
					{
						if (LogLines.Num() > 0)
						{
							const FString Path = FPaths::ProjectSavedDir() / TEXT("PrototypeSWRTShadows") / TEXT("SWRTTrace.log");
							static bool bTruncatedThisSession = false;
							const bool bAppend = bTruncatedThisSession;
							bTruncatedThisSession = true;
							int64 ExistingSize = bAppend ? 0 : (1024 * 1024 + 1);
							FString Block = FString::Printf(
								TEXT("==== %s frame %u ====\n"),
								*FDateTime::Now().ToString(TEXT("%Y.%m.%d-%H.%M.%S")),
								GFrameNumberRenderThread);
							for (const FString& Line : LogLines)
							{
								Block += Line + TEXT("\n");
							}
							Block += TEXT("\n");
							FFileHelper::SaveStringToFile(
								Block, *Path, FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(),
								ExistingSize > 1024 * 1024 ? FILEWRITE_None : FILEWRITE_Append);
							const FString LatestPath =
								FPaths::ProjectSavedDir() / TEXT("PrototypeSWRTShadows") / TEXT("SWRTTraceLatest.log");
							FFileHelper::SaveStringToFile(
								Block, *LatestPath,
								FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(),
								FILEWRITE_None);
						}
					};
					const double CyclesToMs = FPlatformTime::GetSecondsPerCycle() * 1000.0;
					Emit(FString::Printf(
						TEXT("  THREADS ms: game %.2f | draw %.2f (wait %.2f, work %.2f) | rhit %.2f | gpu %.2f")
						TEXT("  <- draw ~= wait means the render thread is blocked on the GPU, not busy"),
						GGameThreadTime * CyclesToMs,
						GRenderThreadTime * CyclesToMs,
						GRenderThreadWaitTime * CyclesToMs,
						(GRenderThreadTime > GRenderThreadWaitTime
							? (GRenderThreadTime - GRenderThreadWaitTime) : 0u) * CyclesToMs,
						GRHIThreadTime * CyclesToMs,
						RHIGetGPUFrameCycles() * CyclesToMs));
					struct FTopShape
					{
						int32 Internal = 0;
						int32 Leaves = 0;
						int32 Children[9] = { 0 };
						int32 LeafInstances = 0;
						int32 WidestLeaf = 0;
						double Branching = 0.0;
						double MeanLeafInstances = 0.0;
					} TopShape;
					{
						int32 Edges = 0;
						for (const FUintVector4& Node : InstanceTree.Nodes)
						{
							const uint32 Count = (Node.W >> GNodeCountShift) & GNodeCountMask;
							if ((Node.W & GNodeLeafBit) != 0)
							{
								++TopShape.Leaves;
								TopShape.LeafInstances += (int32)Count;
								TopShape.WidestLeaf = FMath::Max(TopShape.WidestLeaf, (int32)Count);
							}
							else
							{
								++TopShape.Internal;
								Edges += (int32)Count;
								TopShape.Children[FMath::Clamp((int32)Count, 0, 8)] += 1;
							}
						}
						TopShape.Branching = TopShape.Internal > 0
							? (double)Edges / (double)TopShape.Internal : 0.0;
						TopShape.MeanLeafInstances = TopShape.Leaves > 0
							? (double)TopShape.LeafInstances / (double)TopShape.Leaves : 0.0;
					}
					Emit(FString::Printf(
						TEXT("SWRT.Trace stats: %u rays | per ray: %.1f nodes, ")
						TEXT("%.1f bounds tests, %.1f triangle tests | %d sources, %u tris, %d nodes, %u roots, leaf %d arity %d")
						TEXT(" | TOP LEVEL per ray: %.1f nodes, %.1f bounds tests | LEAF boxes per ray: %.1f")
						TEXT(" | builds this frame %u, casters %d -> instances %u, instance nodes %d, top depth %d")
						TEXT(" | INSTANCE BOX SLACK: mean %.2fx tight, max %.2fx, over %d")
						TEXT(" | INSTANCES per ray: %.2f reached, %.2f past the world box, %.2f entered")
						TEXT(" (TightRoot rejects %.1f%%),")
						TEXT(" %.1f top-level box tests each")
						TEXT(" | TOP LEVEL shape: %d internal + %d leaves, branching %.3f,")
						TEXT(" children 2:%d 3:%d 4:%d, leaves hold %.2f instances mean, %d widest")
						TEXT(" | LEAF box reject %.1f%% | tri box/area mean %.1f max %.1f over %d sorted")
						TEXT(" | STACK high water: blas %u of %d, instance %u of %d, OVERFLOWS %u")
						TEXT(" | SAH order %d of %d slots, %d culled by feedback, %d frames grown")
						TEXT(" | CONFIG tilelist %u group %ux%u swizzle %u rect %dx%d"),
						Counts[0],
						(double)Counts[1] / Rays,
						(double)Counts[2] / Rays,
						(double)Counts[3] / Rays,
						Gathered.Sources.Num(), Gathered.TotalTriangles, Layout.Descs.Num(), NumRoots,
						GLeafTriangles,
						(int32)GNodeArity,
						(double)Counts[4] / Rays,
						(double)Counts[5] / Rays,
						(double)Counts[6] / Rays,
						BuildsThisFrame, Casters.Num(), NumInstances, InstanceTree.Nodes.Num(), InstanceTree.Depth,
						BoundsSlackCount > 0
							? BoundsSlackSum / (double)BoundsSlackCount : 0.0,
						BoundsSlackMax,
						BoundsSlackCount,
						(double)Counts[10] / Rays,
						(double)Counts[38] / Rays,
						(double)Counts[11] / Rays,
						Counts[38] > 0
							? 100.0 * (1.0 - (double)Counts[11] / (double)Counts[38])
							: 0.0,
						Counts[10] > 0
							? (double)Counts[5] / (double)Counts[10]
							: 0.0,
						TopShape.Internal, TopShape.Leaves, TopShape.Branching,
						TopShape.Children[2], TopShape.Children[3], TopShape.Children[4],
						TopShape.MeanLeafInstances, TopShape.WidestLeaf,
						Counts[6] > 0
							? 100.0 * (1.0 - (double)Counts[59] / (double)Counts[6])
							: 0.0,
						GTriangleCache.ShapeProbe.Mean(),
						GTriangleCache.ShapeProbe.MaxRatio,
						(int32)GTriangleCache.ShapeProbe.Count,
						Counts[7],
						GStackCapacity(false),
						Counts[8],
						GStackCapacity(true),
						Counts[9],
						NumOrdersReady, GTriangleCache.Keys.Num(), NumFeedbackCulled, NumFramesGrown,
						GTileListStride,
						FSWRTTraceCS::GetGroupSize().X,
						FSWRTTraceCS::GetGroupSize().Y,
						Parameters->SWRTSwizzleTileWidth,
						TraceRect.Width(), TraceRect.Height()));
					{
						const double Groups = (double)FMath::Max(Counts[14], 1u);
						const double LiveLanes = (double)FMath::Max(Counts[16], 1u);
						const double MeanGroupMax = (double)Counts[13] / Groups;
						const double MeanWork = (double)Counts[15] / LiveLanes;
						const double Divergence = MeanWork > 0.0 ? MeanGroupMax / MeanWork : 0.0;
						const double SharedTop = (double)Counts[17] / Groups;
						const double MaxTop = (double)Counts[18] / Groups;
						const double MeanTop = (double)Counts[5] / Rays;
						Emit(FString::Printf(
							TEXT("  SWRT.Trace DISTRIBUTION over %d groups of %d")
							TEXT(" (%.1f live lanes mean)")
							TEXT(" | WORK per ray: mean %.1f, group max %.1f, DIVERGENCE %.2fx")
							TEXT(" | TOP LEVEL boxes: group min %.1f, mean %.1f, group max %.1f")
							TEXT(" -> %.0f%% of the top level is SHARED by the whole tile")
							TEXT(" | groups that reached any instance: %u of %u (%.1f%%)"),
							Counts[14],
							FSWRTTraceCS::GetGroupSize().X
								* FSWRTTraceCS::GetGroupSize().Y,
							LiveLanes / Groups,
							MeanWork, MeanGroupMax, Divergence,
							SharedTop, MeanTop, MaxTop,
							MeanTop > 0.0 ? 100.0 * SharedTop / MeanTop : 0.0,
							Counts[27], Counts[14],
							100.0 * (double)Counts[27] / Groups));
						{
							const double Warps = FMath::Max((double)Counts[51], 1.0);
							const double WarpMax = (double)Counts[52] / Warps;
							const double WarpWork = (double)Counts[53] / Warps;
							const double WarpLive = (double)Counts[54] / Warps;
							const double LaneEff = WarpMax > 0.0
								? WarpWork / (WarpMax * 32.0) : 0.0;
							const double LiveShare = WarpLive / 32.0;
							const double FinishEarly = (WarpMax > 0.0 && WarpLive > 0.0)
								? WarpWork / (WarpMax * WarpLive) : 0.0;
							Emit(FString::Printf(
								TEXT("  SWRT.Trace WARPS %u")
								TEXT(" | live lanes %.1f of 32 (%.0f%%)")
								TEXT(" | work mean %.1f, warp max %.1f")
								TEXT(" -> LANE EFFICIENCY %.0f%% = %.0f%% live x %.0f%% finish-together")
								TEXT(" | coherent %u hit-only + %u miss-only, MIXED %u (%.0f%% - the only part a ray sort could improve)"),
								Counts[51],
								WarpLive, 100.0 * LiveShare,
								WarpWork / FMath::Max(WarpLive, 1.0), WarpMax,
								100.0 * LaneEff, 100.0 * LiveShare, 100.0 * FinishEarly,
								Counts[55], Counts[56], Counts[57],
								100.0 * (double)Counts[57] / Warps));
							const TCHAR* StageNames[6] = {
								TEXT("out of rect"), TEXT("no depth"), TEXT("out of range"),
								TEXT("out of cone"), TEXT("degenerate"), TEXT("TRACED") };
							uint32 Dispatched = 0;
							for (int32 Stage = 0; Stage < 6; ++Stage)
							{
								Dispatched += Counts[39 + Stage];
							}
							FString Funnel;
							for (int32 Stage = 0; Stage < 6; ++Stage)
							{
								Funnel += FString::Printf(TEXT("%s%s %u (%.0f%%)"),
									Stage > 0 ? TEXT(", ") : TEXT(""),
									StageNames[Stage], Counts[39 + Stage],
									Dispatched > 0
										? 100.0 * (double)Counts[39 + Stage] / (double)Dispatched
										: 0.0);
							}
							Emit(FString::Printf(
								TEXT("  SWRT.Trace FUNNEL over %u threads: %s"),
								Dispatched, *Funnel));
							const double Hits = (double)Counts[46];
							const double Misses = FMath::Max(Rays - Hits, 0.0);
							const double HitWork = (double)Counts[47];
							const double MissWork = (double)Counts[48];
							const double AllWork = FMath::Max(HitWork + MissWork, 1.0);
							Emit(FString::Printf(
								TEXT("  SWRT.Trace OUTCOME: HIT %.0f rays (%.0f%%)")
								TEXT(" costing %.1f nodes and %.1f tris each")
								TEXT(" | MISS %.0f rays (%.0f%%) costing %.1f nodes and %.1f tris each")
								TEXT(" -> misses are %.0f%% of the rays and %.0f%% OF THE WORK"),
								Hits, 100.0 * Hits / FMath::Max(Rays, 1.0),
								Hits > 0.0 ? HitWork / Hits : 0.0,
								Hits > 0.0 ? (double)Counts[49] / Hits : 0.0,
								Misses, 100.0 * Misses / FMath::Max(Rays, 1.0),
								Misses > 0.0 ? MissWork / Misses : 0.0,
								Misses > 0.0 ? (double)Counts[50] / Misses : 0.0,
								100.0 * Misses / FMath::Max(Rays, 1.0),
								100.0 * MissWork / AllWork));
						}
						const double UnionMean = (double)Counts[28] / Groups;
						Emit(FString::Printf(
							TEXT("  SWRT.Trace TILE INSTANCE UNION: mean %.1f distinct")
							TEXT(" instances per 8x8 tile, max %u")
							TEXT(" | the main trace would pay that INSTEAD of %.1f top level box tests")
							TEXT(" -> total box tests %.1f -> %.1f (%.0f%% of today)")
							TEXT(" | %s"),
							UnionMean,
							Counts[29],
							MeanTop,
							MeanTop + (double)Counts[2] / Rays,
							UnionMean + (double)Counts[2] / Rays,
							MeanTop + (double)Counts[2] / Rays > 0.0
								? 100.0 * (UnionMean + (double)Counts[2] / Rays)
									/ (MeanTop + (double)Counts[2] / Rays)
								: 0.0,
							Counts[29] >= 1024u
								? TEXT("SATURATED - more than 1024 instances, the mask wrapped and this UNDERCOUNTS")
								: TEXT("mask not saturated, the count is exact")));
						const TCHAR* BucketNames[8] = {
							TEXT("0"), TEXT("1-4"), TEXT("5-8"), TEXT("9-16"),
							TEXT("17-32"), TEXT("33-64"), TEXT("65-128"), TEXT("129+") };
						FString Histogram;
						for (int32 Bucket = 0; Bucket < 8; ++Bucket)
						{
							Histogram += FString::Printf(
								TEXT("  %s:%u(%.1f%%)"),
								BucketNames[Bucket], Counts[19 + Bucket],
								100.0 * (double)Counts[19 + Bucket] / Rays);
						}
						Emit(FString::Printf(
							TEXT("  SWRT.Trace WORK HISTOGRAM, nodes visited per ray:%s"),
							*Histogram));
						if (Counts[30] + Counts[31] + Counts[32] + Counts[33]
							+ Counts[34] + Counts[35] + Counts[36] + Counts[37] > 0)
						{
							const TCHAR* ListNames[8] = {
								TEXT("0"), TEXT("1-2"), TEXT("3-4"), TEXT("5-8"),
								TEXT("9-16"), TEXT("17-32"), TEXT("33-64"), TEXT("65+") };
							FString ListHistogram;
							for (int32 Bucket = 0; Bucket < 8; ++Bucket)
							{
								ListHistogram += FString::Printf(
									TEXT("  %s:%u(%.1f%%)"),
									ListNames[Bucket], Counts[30 + Bucket],
									100.0 * (double)Counts[30 + Bucket] / Groups);
							}
							Emit(FString::Printf(
								TEXT("  SWRT.Trace LIST HISTOGRAM, instances handed to a tile:%s"),
								*ListHistogram));
						}
					}
					Emit(FString::Printf(
						TEXT("  SWRT.Trace frame: build min (%.1f %.1f %.1f) trace min (%.1f %.1f %.1f) ")
						TEXT("size (%.1f %.1f %.1f) translation (%.1f %.1f %.1f)"),
						BuildSceneMin.X, BuildSceneMin.Y, BuildSceneMin.Z,
						SceneMin.X, SceneMin.Y, SceneMin.Z,
						FrameSize.X, FrameSize.Y, FrameSize.Z,
						Translation.X, Translation.Y, Translation.Z));
					int32 NumDirty = 0;
					for (bool bDirty : DirtySources) { NumDirty += bDirty ? 1 : 0; }
					Emit(FString::Printf(
						TEXT("  SWRT.Trace extraction: %d of %d visible re-extracted, superset %d slots, %u tris"),
						NumDirty, DirtySources.Num(), GTriangleCache.Keys.Num(), GTriangleCache.TotalTriangles));
					Emit(FString::Printf(
						TEXT("  SWRT.Trace CPU ms: total %.3f | gather %.3f, dedup %.3f, slots %.3f, ")
						TEXT("layout %.3f%s, refitlist %.3f, instances %.3f"),
						Timing.Total * 1000.0, Timing.Gather * 1000.0, Timing.Dedup * 1000.0,
						Timing.Slots * 1000.0, Timing.Layout * 1000.0,
						Timing.bRebuiltLayout ? TEXT(" REBUILT") : TEXT(""),
						Timing.RefitList * 1000.0, Timing.Instances * 1000.0));
					int32 NumSkinnedSources = 0;
					for (const FTriangleSource& S : Gathered.Sources) { NumSkinnedSources += S.bSkinned ? 1 : 0; }
					Emit(FString::Printf(TEXT("  SWRT.Trace sources: %d skinned, %d static"),
						NumSkinnedSources, Gathered.Sources.Num() - NumSkinnedSources));
				}
				GStatsReadback->Unlock();
			}
			if (bReadbackTurn && !GStatsReadbackInFlight)
			{
				AddEnqueueCopyPass(GraphBuilder, GStatsReadback.Get(), StatsBuffer, 64 * sizeof(uint32));
				GStatsReadbackInFlight = true;
				GStatsReadbackFrame = GFrameNumberRenderThread;
			}
		}
		return MaskTexture;
	}

}

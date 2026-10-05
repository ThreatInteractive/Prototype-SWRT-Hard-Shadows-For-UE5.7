// SPDX-License-Identifier: Apache-2.0
//
// Which primitives this light treats as shadow casters, and therefore what the
// BVH is built from.
//
// A caster only matters if its shadow can land somewhere the player can see.
// Everything else is work spent on a shadow nobody will ever look at, so the
// gather keeps a primitive only when it sits inside BOTH of these:
//
//   the CONE     everything the light can reach. 
//   the HULL     the convex volume wrapping the bulb together with the eight
//                corners of the camera's view, so it is the bulb joined to
//                everything on screen. A caster that shadows a visible point
//                has to lie on the segment between the two, which puts it
//                inside this volume.
//
// The camera's view is cut short at the distance the light stops reaching, so
// the hull is a squashed box rather than the whole view, which is what makes
// it tight enough to reject anything.
//
// Both tests are conservative, so the gather can keep a caster a sharper test
// would drop, but it can never drop one that matters. And the hull follows the
// camera, so a caster is readmitted the moment the view turns to put it
// between the bulb and something on screen.

#include "SWRTSpotLightRendering.h"

#include "ScenePrivate.h"
#include "SceneRendering.h"
#include "PrimitiveSceneInfo.h"
#include "PrimitiveSceneProxy.h"
#include "UnrealEngine.h"
#include "ConvexVolume.h"

namespace
{
	TAutoConsoleVariable<int32> CVarCasterCull(
		TEXT("r.SWRTSpotLight.CasterCull"),
		1,
		TEXT("Cull casters against the visible region as well as the cone.\n")
		TEXT("  0  the cone frustum alone, which admits everything the light can reach.\n")
		TEXT("  1  on (default): the cone intersected with hull(light, camera frustum)."),
		ECVF_RenderThreadSafe);

	TAutoConsoleVariable<int32> CVarCasterLog(
		TEXT("r.SWRTSpotLight.CasterLog"),
		0,
		TEXT("Log the caster gather.\n")
		TEXT("  1  counts a light a frame.\n")
		TEXT("  2  name every primitive and why it was kept or rejected."),
		ECVF_RenderThreadSafe);

	float GetEngineCVarFloat(const TCHAR* Name, float Default)
	{
		const IConsoleVariable* Variable = IConsoleManager::Get().FindConsoleVariable(Name);
		return Variable != nullptr ? Variable->GetFloat() : Default;
	}

	FString DescribePrimitive(const FPrimitiveSceneProxy* Proxy)
	{
		return FString::Printf(TEXT("%s/%s"),
			*Proxy->GetOwnerName().ToString(), *Proxy->GetResourceName().ToString());
	}

	/** The eight corners of a perspective frustum, near face then far face. */
	void BuildFrustumCorners(
		const FVector& Origin,
		const FVector& Forward,
		const FVector& Right,
		const FVector& Up,
		double TanHalfX,
		double TanHalfY,
		double NearDistance,
		double FarDistance,
		FVector (&OutCorners)[8])
	{
		const double Distances[2] = { NearDistance, FarDistance };
		for (int32 FaceIndex = 0; FaceIndex < 2; ++FaceIndex)
		{
			const double Distance = Distances[FaceIndex];
			const FVector Centre = Origin + Forward * Distance;
			const FVector HalfRight = Right * (TanHalfX * Distance);
			const FVector HalfUp = Up * (TanHalfY * Distance);

			OutCorners[FaceIndex * 4 + 0] = Centre - HalfRight - HalfUp;
			OutCorners[FaceIndex * 4 + 1] = Centre + HalfRight - HalfUp;
			OutCorners[FaceIndex * 4 + 2] = Centre - HalfRight + HalfUp;
			OutCorners[FaceIndex * 4 + 3] = Centre + HalfRight + HalfUp;
		}
	}

	/**
	 * Brute force convex hull of a small point set, as inward facing planes.
	 * Every triple of points is a candidate face and is kept when the whole set
	 * lies on one side of it. Nine points here, so the cubic loop is 84
	 * candidates.
	 */
	void BuildConvexHullPlanes(TConstArrayView<FVector> Points, TArray<FPlane>& OutPlanes)
	{
		OutPlanes.Reset();

		const int32 Num = Points.Num();
		if (Num < 4)
		{
			return;
		}

		const double Epsilon = 0.1;

		for (int32 i = 0; i < Num; ++i)
		{
			for (int32 j = i + 1; j < Num; ++j)
			{
				for (int32 k = j + 1; k < Num; ++k)
				{
					FVector Normal = FVector::CrossProduct(Points[j] - Points[i], Points[k] - Points[i]);
					const double Length = Normal.Size();
					if (Length < UE_DOUBLE_SMALL_NUMBER)
					{
						continue;
					}
					Normal /= Length;

					double W = FVector::DotProduct(Normal, Points[i]);

					int32 NumAbove = 0;
					int32 NumBelow = 0;
					for (int32 m = 0; m < Num; ++m)
					{
						const double Distance = FVector::DotProduct(Normal, Points[m]) - W;
						if (Distance > Epsilon) { ++NumAbove; }
						else if (Distance < -Epsilon) { ++NumBelow; }
					}

					if (NumAbove > 0 && NumBelow > 0)
					{
						continue;
					}

					if (NumAbove > 0)
					{
						Normal = -Normal;
						W = -W;
					}

					const FPlane Candidate(Normal, W);

					bool bDuplicate = false;
					for (const FPlane& Existing : OutPlanes)
					{
						if (FVector::DotProduct(FVector(Existing), Normal) > 0.9999
							&& FMath::Abs(Existing.W - W) < Epsilon)
						{
							bDuplicate = true;
							break;
						}
					}

					if (!bDuplicate)
					{
						OutPlanes.Add(Candidate);
					}
				}
			}
		}
	}

	/** Clips a segment to a clip volume. False when it lies entirely outside. */
	bool ClipSegmentToClipVolume(const FMatrix& WorldToClip, FVector& InOutStart, FVector& InOutEnd)
	{
		const FVector Start = InOutStart;
		const FVector End = InOutEnd;

		const FVector4 ClipStart = WorldToClip.TransformFVector4(FVector4(Start, 1.0));
		const FVector4 ClipEnd = WorldToClip.TransformFVector4(FVector4(End, 1.0));

		const double BoundsStart[6] =
		{
			ClipStart.W + ClipStart.X, ClipStart.W - ClipStart.X,
			ClipStart.W + ClipStart.Y, ClipStart.W - ClipStart.Y,
			ClipStart.Z,               ClipStart.W - ClipStart.Z,
		};
		const double BoundsEnd[6] =
		{
			ClipEnd.W + ClipEnd.X, ClipEnd.W - ClipEnd.X,
			ClipEnd.W + ClipEnd.Y, ClipEnd.W - ClipEnd.Y,
			ClipEnd.Z,             ClipEnd.W - ClipEnd.Z,
		};

		double MinT = 0.0;
		double MaxT = 1.0;

		for (int32 Index = 0; Index < 6; ++Index)
		{
			const double AtStart = BoundsStart[Index];
			const double AtEnd = BoundsEnd[Index];

			if (AtStart < 0.0 && AtEnd < 0.0)
			{
				return false;
			}
			if (AtStart >= 0.0 && AtEnd >= 0.0)
			{
				continue;
			}

			const double Crossing = AtStart / (AtStart - AtEnd);
			if (AtStart < 0.0)
			{
				MinT = FMath::Max(MinT, Crossing);
			}
			else
			{
				MaxT = FMath::Min(MaxT, Crossing);
			}

			if (MinT > MaxT)
			{
				return false;
			}
		}

		InOutStart = Start + (End - Start) * MinT;
		InOutEnd = Start + (End - Start) * MaxT;
		return true;
	}

	void AppendFrustumEdges(const FVector (&Corners)[8], TArray<TPair<FVector, FVector>, TInlineAllocator<24>>& OutEdges)
	{
		static const int32 EdgeIndices[12][2] =
		{
			{0,1},{1,3},{3,2},{2,0},
			{4,5},{5,7},{7,6},{6,4},
			{0,4},{1,5},{2,6},{3,7},
		};

		for (int32 Index = 0; Index < 12; ++Index)
		{
			OutEdges.Emplace(Corners[EdgeIndices[Index][0]], Corners[EdgeIndices[Index][1]]);
		}
	}

	/**
	 * The part of the cone the camera can see, as a point set in translated
	 * world space. Built by clipping each frustum's edges against the other's
	 * clip volume, which is what makes it the intersection rather than either
	 * bound on its own.
	 *
	 * Invalid means the cone is entirely outside the camera frustum, in which
	 * case there is nothing to shadow and no caster is needed.
	 */
	struct FVisibleRegion
	{
		double MaxViewDepth = 0.0;
		bool bValid = false;
	};

	FVisibleRegion ComputeVisibleRegion(
		const FViewInfo& View,
		const FSWRTSpotLightRenderData& Light,
		const FMatrix& TranslatedWorldToConeClip,
		double ConeNear,
		double ConeFar)
	{
		FVisibleRegion Region;

		const FVector PreViewTranslation = View.ViewMatrices.GetPreViewTranslation();
		const FMatrix& TranslatedWorldToCameraClip = View.ViewMatrices.GetTranslatedViewProjectionMatrix();
		const FMatrix& TranslatedWorldToView = View.ViewMatrices.GetTranslatedViewMatrix();
		const FMatrix InverseView = View.ViewMatrices.GetInvViewMatrix();

		const FVector ViewOrigin = View.ViewMatrices.GetViewOrigin() + PreViewTranslation;
		const FVector ViewRight = InverseView.GetUnitAxis(EAxis::X);
		const FVector ViewUp = InverseView.GetUnitAxis(EAxis::Y);
		const FVector ViewForward = InverseView.GetUnitAxis(EAxis::Z);

		const FMatrix& Projection = View.ViewMatrices.GetProjectionMatrix();
		const double TanHalfX = Projection.M[0][0] != 0.0 ? 1.0 / Projection.M[0][0] : 1.0;
		const double TanHalfY = Projection.M[1][1] != 0.0 ? 1.0 / Projection.M[1][1] : 1.0;

		const FVector LightBoundsCentre = Light.Bounds.Center + PreViewTranslation;
		const double CameraNear = FMath::Max((double)View.NearClippingDistance, 1.0);
		const double CameraFar = FMath::Max(
			CameraNear * 1.01,
			(double)(LightBoundsCentre - ViewOrigin).Size() + (double)Light.Bounds.W);

		FVector CameraCorners[8];
		BuildFrustumCorners(ViewOrigin, ViewForward, ViewRight, ViewUp, TanHalfX, TanHalfY, CameraNear, CameraFar, CameraCorners);

		const FVector LightPosition = Light.Position + PreViewTranslation;
		const FVector LightRight = Light.LightToWorld.GetUnitAxis(EAxis::Y);
		const FVector LightUp = Light.LightToWorld.GetUnitAxis(EAxis::Z);
		const double TanOuter = FMath::Tan((double)Light.OuterConeAngle);

		FVector LightCorners[8];
		BuildFrustumCorners(LightPosition, Light.Direction, LightRight, LightUp, TanOuter, TanOuter, ConeNear, ConeFar, LightCorners);

		TArray<TPair<FVector, FVector>, TInlineAllocator<24>> CameraEdges;
		TArray<TPair<FVector, FVector>, TInlineAllocator<24>> LightEdges;
		AppendFrustumEdges(CameraCorners, CameraEdges);
		AppendFrustumEdges(LightCorners, LightEdges);

		int32 NumPoints = 0;
		double MaxDepth = -UE_DOUBLE_BIG_NUMBER;

		auto AccumulatePoint = [&](const FVector& Point)
		{
			const FVector4 CameraClip = TranslatedWorldToCameraClip.TransformFVector4(FVector4(Point, 1.0));
			if (CameraClip.W <= UE_SMALL_NUMBER)
			{
				return;
			}

			MaxDepth = FMath::Max(MaxDepth, TranslatedWorldToView.TransformPosition(Point).Z);
			++NumPoints;
		};

		for (const TPair<FVector, FVector>& Edge : CameraEdges)
		{
			FVector Start = Edge.Key;
			FVector End = Edge.Value;
			if (ClipSegmentToClipVolume(TranslatedWorldToConeClip, Start, End))
			{
				AccumulatePoint(Start);
				AccumulatePoint(End);
			}
		}

		for (const TPair<FVector, FVector>& Edge : LightEdges)
		{
			FVector Start = Edge.Key;
			FVector End = Edge.Value;
			if (ClipSegmentToClipVolume(TranslatedWorldToCameraClip, Start, End))
			{
				AccumulatePoint(Start);
				AccumulatePoint(End);
			}
		}

		if (NumPoints < 4)
		{
			return Region;
		}

		Region.MaxViewDepth = MaxDepth;
		Region.bValid = true;
		return Region;
	}
	/**
	 * Why one primitive did or did not become a caster.
	 *
	 * Ordered: everything from HiddenInView down was near enough and pointed at
	 * by the light, and only failed for a reason about the VIEW rather than
	 * about the light. CountsAsReaching splits the two halves, which is what
	 * the log reports.
	 */
	enum class ECasterVerdict : uint8
	{
		Keep = 0,

		OutOfLightRange,
		DoesNotCastDynamicShadow,
		NaniteMesh,
		SelfShadowOnly,
		OutsideCullVolume,

		HiddenInView,
		BeyondDrawDistance,
		NotOpaqueOrMasked,
		NotShadowRelevant,
		TooSmallOnScreen,
		NothingDrawable,
	};

	bool CountsAsReaching(ECasterVerdict Verdict)
	{
		return Verdict == ECasterVerdict::Keep || Verdict >= ECasterVerdict::HiddenInView;
	}

	/** The verdicts worth a line. The rest are ordinary and would drown the log. */
	bool IsWorthLogging(ECasterVerdict Verdict)
	{
		switch (Verdict)
		{
		case ECasterVerdict::DoesNotCastDynamicShadow:
		case ECasterVerdict::NaniteMesh:
		case ECasterVerdict::SelfShadowOnly:
		case ECasterVerdict::OutsideCullVolume:
		case ECasterVerdict::NotOpaqueOrMasked:
		case ECasterVerdict::NotShadowRelevant:
		case ECasterVerdict::TooSmallOnScreen:
			return true;
		default:
			return false;
		}
	}

	const TCHAR* DescribeVerdict(ECasterVerdict Verdict)
	{
		switch (Verdict)
		{
		case ECasterVerdict::Keep:                     return TEXT("kept");
		case ECasterVerdict::OutOfLightRange:          return TEXT("out of the light's range");
		case ECasterVerdict::DoesNotCastDynamicShadow: return TEXT("does not cast a dynamic shadow");
		case ECasterVerdict::NaniteMesh:               return TEXT("nanite, which this path cannot read");
		case ECasterVerdict::SelfShadowOnly:           return TEXT("casts a self shadow only");
		case ECasterVerdict::OutsideCullVolume:        return TEXT("outside the cone and the hull");
		case ECasterVerdict::HiddenInView:             return TEXT("hidden in this view");
		case ECasterVerdict::BeyondDrawDistance:       return TEXT("beyond its own draw distance");
		case ECasterVerdict::NotOpaqueOrMasked:        return TEXT("neither opaque nor masked");
		case ECasterVerdict::NotShadowRelevant:        return TEXT("not shadow relevant");
		case ECasterVerdict::TooSmallOnScreen:         return TEXT("too small on screen to shadow with");
		default:                                       return TEXT("nothing drawable");
		}
	}

	/** Everything the classifier needs that is the same for every primitive. */
	struct FCasterGatherContext
	{
		const FViewInfo* View = nullptr;
		const FScene* Scene = nullptr;
		const FSWRTSpotLightRenderData* Light = nullptr;
		const FConvexVolume* CullVolume = nullptr;

		FVector ShadowViewOrigin = FVector::ZeroVector;
		float MinScreenRadius = 0.0f;
		float LODScaleSquared = 1.0f;
	};

	/**
	 * One primitive, start to finish. bOutStatic says which half of the engine
	 * it would be drawn from, which only means anything when the verdict is
	 * Keep.
	 */
	ECasterVerdict ClassifyCaster(const FCasterGatherContext& Context, int32 PrimitiveIndex, bool& bOutStatic)
	{
		bOutStatic = false;

		const FScene& Scene = *Context.Scene;
		const FViewInfo& View = *Context.View;
		const FSWRTSpotLightRenderData& Light = *Context.Light;

		const FPrimitiveSceneInfo* SceneInfo = Scene.Primitives[PrimitiveIndex];
		const FPrimitiveSceneProxy* Proxy = Scene.PrimitiveSceneProxies[PrimitiveIndex];
		if (SceneInfo == nullptr || Proxy == nullptr)
		{
			return ECasterVerdict::OutOfLightRange;
		}

		const FBoxSphereBounds& Bounds = Scene.PrimitiveBounds[PrimitiveIndex].BoxSphereBounds;

		// ---- Can the light touch it at all
		{
			const double Reach = (double)Bounds.SphereRadius + Light.Bounds.W;
			if (FVector::DistSquared(Bounds.Origin, Light.Bounds.Center) > FMath::Square(Reach))
			{
				return ECasterVerdict::OutOfLightRange;
			}
		}

		// ---- Is it allowed to cast
		{
			const FPrimitiveFlagsCompact& Flags = Scene.PrimitiveFlagsCompact[PrimitiveIndex];

			if (!Flags.bCastDynamicShadow)
			{
				return ECasterVerdict::DoesNotCastDynamicShadow;
			}
			if (Flags.bIsNaniteMesh)
			{
				return ECasterVerdict::NaniteMesh;
			}
			if (Proxy->CastsSelfShadowOnly())
			{
				return ECasterVerdict::SelfShadowOnly;
			}
		}

		// ---- Can its shadow land anywhere on screen
		if (!Context.CullVolume->IntersectBox(Bounds.Origin, Bounds.BoxExtent))
		{
			return ECasterVerdict::OutsideCullVolume;
		}

		// Past here the primitive reaches the light, and anything that stops it
		// now is about this view rather than about the light.

		// ---- Is it shown in this view
		{
			const FPrimitiveComponentId ComponentId = Proxy->GetPrimitiveComponentId();

			const bool bFilteredOut =
				View.ShowOnlyPrimitives.IsSet() && !View.ShowOnlyPrimitives->Contains(ComponentId);

			if (bFilteredOut || View.HiddenPrimitives.Contains(ComponentId))
			{
				return ECasterVerdict::HiddenInView;
			}
		}

		const float DistanceSquared = (Bounds.Origin - Context.ShadowViewOrigin).SizeSquared();

		// ---- Relevance, reused from visibility when that already ran
		FPrimitiveViewRelevance Relevance = View.PrimitiveViewRelevanceMap[PrimitiveIndex];
		if (!Relevance.bInitializedThisFrame)
		{
			// Visibility did not reach it, so its draw distances have not been
			// applied yet either.
			if (View.IsPerspectiveProjection())
			{
				const float MaxDrawDistance =
					Proxy->GetMaxDrawDistance() * GetCachedScalabilityCVars().ViewDistanceScale;
				const float MinDrawDistance = Proxy->GetMinDrawDistance();

				const bool bTooFar = MaxDrawDistance > 0.0f && DistanceSquared > FMath::Square(MaxDrawDistance);
				const bool bTooNear = DistanceSquared < FMath::Square(MinDrawDistance);

				if (bTooFar || bTooNear)
				{
					return ECasterVerdict::BeyondDrawDistance;
				}
			}

			Relevance = Proxy->GetViewRelevance(&View);
		}

		if (!Relevance.bOpaque && !Relevance.bMasked)
		{
			return ECasterVerdict::NotOpaqueOrMasked;
		}
		if (!Relevance.bShadowRelevance)
		{
			return ECasterVerdict::NotShadowRelevant;
		}

		// ---- Which half of the engine would draw it
		bOutStatic = SceneInfo->StaticMeshes.Num() > 0 && Relevance.bStaticRelevance;

		if (bOutStatic)
		{
			// A static caster small enough on screen is not worth the geometry.
			const float ScreenRadiusLimit =
				FMath::Square(Context.MinScreenRadius) * DistanceSquared * Context.LODScaleSquared;

			if (FMath::Square(Bounds.SphereRadius) <= ScreenRadiusLimit)
			{
				return ECasterVerdict::TooSmallOnScreen;
			}

			return ECasterVerdict::Keep;
		}

		return Relevance.bDynamicRelevance ? ECasterVerdict::Keep : ECasterVerdict::NothingDrawable;
	}
}

void SWRTSpot::GatherSWRTCasters(
	FRDGBuilder& GraphBuilder,
	const FViewInfo& View,
	const FSWRTSpotLightRenderData& Light,
	TArray<const FPrimitiveSceneProxy*>& OutCasters)
{
	const FScene* Scene = View.Family != nullptr ? View.Family->Scene->GetRenderScene() : nullptr;
	if (Scene == nullptr)
	{
		return;
	}

	const int32 CasterLogLevel = CVarCasterLog.GetValueOnRenderThread();
	const bool bVerbose = CasterLogLevel >= 2;

	const FVector PreViewTranslation = View.ViewMatrices.GetPreViewTranslation();
	const FVector LightPosition = Light.Position + PreViewTranslation;

	// The cone, as a clip volume. The near distance is deliberately tiny: it
	// only has to keep the projection finite, and anything larger would drop
	// casters standing close to the bulb.
	const double ConeFar = FMath::Max((double)Light.Radius, 1.0);
	const double ConeNear = FMath::Max(1.0, ConeFar * 0.001);

	const FMatrix FaceMatrix(
		FPlane(0, 0, 1, 0),
		FPlane(0, 1, 0, 0),
		FPlane(-1, 0, 0, 0),
		FPlane(0, 0, 0, 1));

	const FMatrix TranslatedWorldToConeView =
		FTranslationMatrix(-LightPosition) * Light.WorldToLightRotation * FaceMatrix;
	const FMatrix ConeViewToClip =
		FPerspectiveMatrix(Light.OuterConeAngle, 1.0f, 1.0f, (float)ConeNear, (float)ConeFar);
	const FMatrix TranslatedWorldToConeClip = TranslatedWorldToConeView * ConeViewToClip;

	FConvexVolume CasterFrustum;
	GetViewFrustumBounds(
		CasterFrustum,
		FTranslationMatrix(PreViewTranslation) * TranslatedWorldToConeClip,
		true);

	if (CVarCasterCull.GetValueOnRenderThread() != 0)
	{
		const FVisibleRegion Region = ComputeVisibleRegion(
			View, Light, TranslatedWorldToConeClip, ConeNear, ConeFar);

		if (!Region.bValid)
		{
			// The cone is entirely outside the camera frustum. Nothing on
			// screen can be shadowed by this light, so there is no caster to
			// gather and no trace to run.
			if (CasterLogLevel >= 1)
			{
				UE_LOG(LogTemp, Display, TEXT("SWRT casters: the cone is outside the camera frustum, nothing gathered."));
			}
			return;
		}

		const FMatrix InvViewMatrix = View.ViewMatrices.GetInvViewMatrix();
		const FVector CameraOrigin = View.ViewMatrices.GetViewOrigin();
		const FVector CameraRight = InvViewMatrix.GetScaledAxis(EAxis::X).GetSafeNormal();
		const FVector CameraUp = InvViewMatrix.GetScaledAxis(EAxis::Y).GetSafeNormal();
		const FVector CameraForward = InvViewMatrix.GetScaledAxis(EAxis::Z).GetSafeNormal();

		const FMatrix& Projection = View.ViewMatrices.GetProjectionMatrix();
		const double TanHalfX = Projection.M[0][0] != 0.0 ? 1.0 / Projection.M[0][0] : 1.0;
		const double TanHalfY = Projection.M[1][1] != 0.0 ? 1.0 / Projection.M[1][1] : 1.0;

		// The frustum is cut off at the region's far depth. Nothing past it is
		// lit by this light, so nothing past it is a receiver anyone would miss.
		const double BoundNear = FMath::Max((double)View.NearClippingDistance, 1.0);
		const double BoundFar = FMath::Max(Region.MaxViewDepth, BoundNear * 1.01);

		FVector Corners[8];
		BuildFrustumCorners(CameraOrigin, CameraForward, CameraRight, CameraUp, TanHalfX, TanHalfY, BoundNear, BoundFar, Corners);

		TArray<FVector, TInlineAllocator<9>> HullPoints;
		HullPoints.Append(Corners, 8);
		HullPoints.Add(Light.Position);

		TArray<FPlane> HullPlanes;
		BuildConvexHullPlanes(HullPoints, HullPlanes);

		if (HullPlanes.Num() >= 4)
		{
			for (const FPlane& Plane : HullPlanes)
			{
				CasterFrustum.Planes.Add(Plane);
			}
			CasterFrustum.Init();

			if (CasterLogLevel >= 1)
			{
				UE_LOG(LogTemp, Display,
					TEXT("SWRT casters: cone plus hull(light, camera frustum), %d hull planes over camera depth %.0f to %.0f"),
					HullPlanes.Num(), BoundNear, BoundFar);
			}
		}
	}

	// ---- Walk the scene once, classifying every primitive
	FCasterGatherContext Context;
	Context.View = &View;
	Context.Scene = Scene;
	Context.Light = &Light;
	Context.CullVolume = &CasterFrustum;
	Context.ShadowViewOrigin = View.GetShadowViewMatrices().GetViewOrigin();
	Context.MinScreenRadius = GetEngineCVarFloat(TEXT("r.Shadow.RadiusThreshold"), 0.01f);
	Context.LODScaleSquared = FMath::Square(View.LODDistanceFactor);

	int32 NumReaching = 0;

	for (int32 PrimitiveIndex = 0; PrimitiveIndex < Scene->Primitives.Num(); ++PrimitiveIndex)
	{
		const FPrimitiveSceneProxy* Proxy = Scene->PrimitiveSceneProxies[PrimitiveIndex];
		if (Proxy == nullptr)
		{
			continue;
		}

		bool bStatic = false;
		const ECasterVerdict Verdict = ClassifyCaster(Context, PrimitiveIndex, bStatic);

		if (CountsAsReaching(Verdict))
		{
			++NumReaching;
		}

		if (Verdict == ECasterVerdict::Keep)
		{
			OutCasters.Add(Proxy);

			if (bVerbose)
			{
				UE_LOG(LogTemp, Log, TEXT("  SWRT caster: %s kept, %s"),
					*DescribePrimitive(Proxy), bStatic ? TEXT("static") : TEXT("dynamic"));
			}
			continue;
		}

		if (bVerbose && IsWorthLogging(Verdict))
		{
			UE_LOG(LogTemp, Log, TEXT("  SWRT caster: %s rejected, %s"),
				*DescribePrimitive(Proxy), DescribeVerdict(Verdict));
		}
	}

	if (CasterLogLevel >= 1)
	{
		UE_LOG(LogTemp, Display, TEXT("SWRT casters: %d could reach the light, %d kept of %d scene primitives"),
			NumReaching, OutCasters.Num(), Scene->Primitives.Num());
	}
}

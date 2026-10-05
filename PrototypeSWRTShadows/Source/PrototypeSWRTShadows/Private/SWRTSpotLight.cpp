// SPDX-License-Identifier: Apache-2.0

#include "SWRTSpotLight.h"

#include "Components/ArrowComponent.h"
#include "EngineUtils.h"
#include "Engine/World.h"
#include "HAL/IConsoleManager.h"
#include "SceneView.h"

// FViewInfo, for the cast in the render hook.
#include "ScenePrivate.h"

namespace
{
	TAutoConsoleVariable<int32> CVarSWRTEnabled(
		TEXT("r.SWRTSpotLight.Enabled"),
		1,
		TEXT("Master switch for the SWRT spot lights. 0 renders none of them."),
		ECVF_RenderThreadSafe);
}

/*=============================================================================
	COMPONENT AND ACTOR
=============================================================================*/
USWRTSpotLightComponent::USWRTSpotLightComponent(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	Mobility = EComponentMobility::Movable;
}

ASWRTSpotLight::ASWRTSpotLight()
{
	PrimaryActorTick.bCanEverTick = false;

	SpotLightComponent = CreateDefaultSubobject<USWRTSpotLightComponent>(TEXT("SpotLightComponent0"));
	RootComponent = SpotLightComponent;
	SpotLightComponent->SetRelativeRotation(FRotator(-90.0f, 0.0f, 0.0f));

#if WITH_EDITORONLY_DATA
	ArrowComponent = CreateEditorOnlyDefaultSubobject<UArrowComponent>(TEXT("ArrowComponent0"));
	if (ArrowComponent)
	{
		ArrowComponent->ArrowColor = SpotLightComponent->GetLightColor().ToFColor(true);
		ArrowComponent->bTreatAsASprite = true;
		ArrowComponent->SetupAttachment(SpotLightComponent);
		ArrowComponent->bLightAttachment = true;
		ArrowComponent->bIsScreenSizeScaled = true;
	}
#endif
}

#if WITH_EDITOR
void ASWRTSpotLight::EditorApplyScale(const FVector& DeltaScale, const FVector* PivotLocation, bool bAltDown, bool bShiftDown, bool bCtrlDown)
{
	const FVector ModifiedScale = DeltaScale * (AActor::bUsePercentageBasedScaling ? 10000.0f : 100.0f);

	if (bCtrlDown)
	{
		FMath::ApplyScaleToFloat(SpotLightComponent->OuterConeAngle, ModifiedScale, 0.01f);
		SpotLightComponent->OuterConeAngle = FMath::Min(89.0f, SpotLightComponent->OuterConeAngle);
		SpotLightComponent->InnerConeAngle = FMath::Min(SpotLightComponent->OuterConeAngle, SpotLightComponent->InnerConeAngle);
	}
	else if (bAltDown)
	{
		FMath::ApplyScaleToFloat(SpotLightComponent->InnerConeAngle, ModifiedScale, 0.01f);
		SpotLightComponent->InnerConeAngle = FMath::Min(89.0f, SpotLightComponent->InnerConeAngle);
		SpotLightComponent->OuterConeAngle = FMath::Max(SpotLightComponent->OuterConeAngle, SpotLightComponent->InnerConeAngle);
	}
	else
	{
		FMath::ApplyScaleToFloat(SpotLightComponent->AttenuationRadius, ModifiedScale);
	}

	PostEditChange();
}

void ASWRTSpotLight::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
	Super::PostEditChangeProperty(PropertyChangedEvent);

	if (ArrowComponent && SpotLightComponent)
	{
		ArrowComponent->ArrowColor = SpotLightComponent->GetLightColor().ToFColor(true);
	}
}
#endif

/*=============================================================================
	GAME THREAD GATHER
=============================================================================*/
namespace
{
	bool ShouldRenderLightComponent(const ULightComponent* Component)
	{
		if (!Component || !Component->IsRegistered() || !Component->bAffectsWorld)
		{
			return false;
		}

		// EV units are allowed to be zero or negative and still emit.
		const bool bValidIntensity = Component->Intensity > 0.0f || Component->GetLightUnits() == ELightUnits::EV;

		return bValidIntensity
			&& Component->ShouldComponentAddToScene()
			&& Component->ShouldRender();
	}

	bool GatherOneLight(const USpotLightComponent* Component, FSWRTSpotLightRenderData& Out)
	{
		if (!ShouldRenderLightComponent(Component))
		{
			return false;
		}

		Out.LightKey = FObjectKey(Component);
		Out.Color = Component->GetColoredLightBrightness();
		if (Out.Color.IsAlmostBlack())
		{
			return false;
		}

		const FTransform& ComponentTransform = Component->GetComponentTransform();
		Out.LightToWorld = ComponentTransform.ToMatrixNoScale();
		Out.WorldToLightRotation = Out.LightToWorld.Inverse().RemoveTranslation();
		Out.Position = ComponentTransform.GetLocation();
		Out.Direction = Out.LightToWorld.GetUnitAxis(EAxis::X);
		Out.Tangent = Out.LightToWorld.GetUnitAxis(EAxis::Z);

		Out.InverseExposureBlend = Component->InverseExposureBlend;
		Out.Radius = FMath::Max(Component->AttenuationRadius, 1.0f);
		Out.FalloffExponent = Component->bUseInverseSquaredFalloff ? 0.0f : Component->LightFalloffExponent;
		Out.SourceRadius = Component->SourceRadius;
		Out.SoftSourceRadius = Component->SoftSourceRadius;
		Out.SourceLength = Component->SourceLength;
		Out.SpecularScale = FMath::Clamp(Component->SpecularScale, 0.0f, 1.0f);
		Out.DiffuseScale = FMath::Clamp(Component->DiffuseScale, 0.0f, 1.0f);
		Out.MaxDrawDistance = Component->MaxDrawDistance;
		Out.MaxDistanceFadeRange = Component->MaxDistanceFadeRange;
		Out.ContactShadowLength = Component->ContactShadowLength * (Component->ContactShadowLengthInWS ? -1.0f : 1.0f);
		Out.ContactShadowCastingIntensity = Component->ContactShadowCastingIntensity;
		Out.ContactShadowNonCastingIntensity = Component->ContactShadowNonCastingIntensity;
		Out.bCastShadows = Component->CastShadows && Component->CastDynamicShadows;

		// r.AllowDepthBoundsTest is not render thread safe, so it is read here.
		static const IConsoleVariable* AllowDepthBoundsTest =
			IConsoleManager::Get().FindConsoleVariable(TEXT("r.AllowDepthBoundsTest"));
		Out.bAllowDepthBoundsTest = !AllowDepthBoundsTest || AllowDepthBoundsTest->GetInt() != 0;

		// FSpotLightSceneProxy's clamps, then the engine's cone bounding sphere.
		const float ClampedInnerConeAngle =
			FMath::Clamp(Component->InnerConeAngle, 0.0f, 89.0f) * (float)UE_PI / 180.0f;
		const float ClampedOuterConeAngle = FMath::Clamp(
			Component->OuterConeAngle * (float)UE_PI / 180.0f,
			ClampedInnerConeAngle + 0.001f,
			89.0f * (float)UE_PI / 180.0f + 0.001f);

		Out.OuterConeAngle = ClampedOuterConeAngle;
		Out.CosOuterCone = FMath::Cos(ClampedOuterConeAngle);
		Out.SinOuterCone = FMath::Sin(ClampedOuterConeAngle);
		Out.InvCosConeDifference = 1.0f / (FMath::Cos(ClampedInnerConeAngle) - Out.CosOuterCone);

		Out.Bounds = FMath::ComputeBoundingSphereForCone(
			Out.Position,
			Out.Direction,
			(FSphere::FReal)Out.Radius,
			(FSphere::FReal)Out.CosOuterCone,
			(FSphere::FReal)Out.SinOuterCone);

		return true;
	}
}

void SWRTSpot::GatherSWRTSpotLights(UWorld* OwningWorld, TArray<FSWRTSpotLightRenderData>& OutLights)
{
	if (!OwningWorld)
	{
		return;
	}

	for (TActorIterator<ASWRTSpotLight> It(OwningWorld); It; ++It)
	{
		FSWRTSpotLightRenderData Light;
		if (GatherOneLight(It->SpotLightComponent, Light))
		{
			OutLights.Add(Light);
		}
	}
}

/*=============================================================================
	VIEW EXTENSION
=============================================================================*/
FSWRTSpotLightViewExtension::FSWRTSpotLightViewExtension(const FAutoRegister& AutoRegister, UWorld* InWorld)
	: FWorldSceneViewExtension(AutoRegister, InWorld)
{
	UE_LOG(LogTemp, Log, TEXT("PrototypeSWRTShadows: view extension registered."));
}

bool FSWRTSpotLightViewExtension::IsActiveThisFrame_Internal(const FSceneViewExtensionContext& Context) const
{
	return CVarSWRTEnabled.GetValueOnAnyThread() != 0
		&& FWorldSceneViewExtension::IsActiveThisFrame_Internal(Context);
}

void FSWRTSpotLightViewExtension::BeginRenderViewFamily(FSceneViewFamily& InViewFamily)
{
	TArray<FSWRTSpotLightRenderData> Gathered;
	SWRTSpot::GatherSWRTSpotLights(GetWorld(), Gathered);

	TSharedRef<FSWRTSpotLightViewExtension, ESPMode::ThreadSafe> SelfRef =
		StaticCastSharedRef<FSWRTSpotLightViewExtension>(AsShared());

	ENQUEUE_RENDER_COMMAND(SWRTSpotLightUpdateLights)(
		[SelfRef, Gathered = MoveTemp(Gathered)](FRHICommandListImmediate&) mutable
		{
			SelfRef->Lights_RenderThread = MoveTemp(Gathered);
		});
}

void FSWRTSpotLightViewExtension::PostRenderBasePassDeferred_RenderThread(
	FRDGBuilder& GraphBuilder,
	FSceneView& InView,
	const FRenderTargetBindingSlots& RenderTargets,
	TRDGUniformBufferRef<FSceneTextureUniformParameters> SceneTextures)
{
	if (Lights_RenderThread.Num() == 0)
	{
		return;
	}

	if (InView.bIsSceneCapture || (InView.Family && InView.Family->bThumbnailRendering))
	{
		return;
	}

	// Under the editor's wireframe view mode slot zero is EditorPrimitiveColor
	// rather than scene colour, so blending into it would tint the wireframe
	// overlay instead of lighting anything.
	if (InView.Family && InView.Family->EngineShowFlags.Wireframe)
	{
		return;
	}

	// RenderTargets arrives as the full base pass render target set: slot zero
	// is scene colour and every slot after it is a GBuffer channel. Reusing the
	// set as given with additive blending would accumulate light into the
	// packed normal, roughness and metallic buffers, so slot zero is pulled out
	// and a fresh single slot binding built from it.
	FRDGTextureRef SceneColorTexture = nullptr;
	int32 SlotIndex = 0;
	RenderTargets.Enumerate([&SceneColorTexture, &SlotIndex](const FRenderTargetBinding& Binding)
		{
			if (SlotIndex == 0)
			{
				SceneColorTexture = Binding.GetTexture();
			}
			++SlotIndex;
		});

	if (!SceneColorTexture)
	{
		return;
	}

	SWRTSpot::RenderSWRTSpotLights(
		GraphBuilder,
		static_cast<FViewInfo&>(InView),
		SceneColorTexture,
		RenderTargets.DepthStencil.GetTexture(),
		Lights_RenderThread);
}

/*=============================================================================
	SUBSYSTEM
=============================================================================*/
void USWRTSpotLightSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	SceneViewExtension = FSceneViewExtensions::NewExtension<FSWRTSpotLightViewExtension>(GetWorld());
}

void USWRTSpotLightSubsystem::Deinitialize()
{
	if (SceneViewExtension.IsValid())
	{
		// The extension is reference counted by the renderer and can outlive
		// this subsystem by a frame, so it is switched off rather than assumed
		// destroyed.
		SceneViewExtension->IsActiveThisFrameFunctions.Empty();

		FSceneViewExtensionIsActiveFunctor IsActiveFunctor;
		IsActiveFunctor.IsActiveFunction = [](const ISceneViewExtension*, const FSceneViewExtensionContext&)
			{
				return TOptional<bool>(false);
			};
		SceneViewExtension->IsActiveThisFrameFunctions.Add(IsActiveFunctor);
		SceneViewExtension.Reset();
	}

	Super::Deinitialize();
}

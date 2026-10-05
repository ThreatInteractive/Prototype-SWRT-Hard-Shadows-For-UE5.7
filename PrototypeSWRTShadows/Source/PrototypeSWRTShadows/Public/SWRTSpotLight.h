// SPDX-License-Identifier: Apache-2.0
//
// The SWRT spot light component and actor, and the view extension that renders
// them.
//
// The component derives from USpotLightComponent so it inherits the whole
// engine property set and, with it, the editor's cone wireframe and drag
// handles, which component visualizers find by walking the class hierarchy.
// CreateSceneProxy returns null, so the light never reaches FScene and the
// engine neither lights nor shadows with it, while the editor still draws it
// because it reads the component directly.
//
// The render hook is PostRenderBasePassDeferred_RenderThread, which fires
// immediately after the base pass, where the engine's own lighting runs.
// Anything added there is still seen by fog, translucency and every later
// stage. The easier hook, PrePostProcessPass_RenderThread, runs after fog and
// translucency are already composited, so light added there floats on top of
// fog and in front of translucent surfaces.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/SpotLightComponent.h"
#include "SceneViewExtension.h"
#include "Subsystems/WorldSubsystem.h"
#include "SWRTSpotLightRendering.h"

#include "SWRTSpotLight.generated.h"

class UArrowComponent;

/**
 * Spot light shadowed by the software ray traced path in this plugin.
 *
 * Every inherited property the plugin implements behaves as it does on an
 * engine spot light. Not implemented, though still shown and saved: IES
 * profiles, light functions, volumetric scattering, lighting channels,
 * translucency lighting and the Lightmass settings.
 */
UCLASS(Blueprintable, ClassGroup = (SWRT), hidecategories = (Object, LightShafts, RayTracing), editinlinenew, meta = (BlueprintSpawnableComponent))
class PROTOTYPESWRTSHADOWS_API USWRTSpotLightComponent : public USpotLightComponent
{
	GENERATED_BODY()

public:
	USWRTSpotLightComponent(const FObjectInitializer& ObjectInitializer);

	/** No proxy, so the engine neither lights nor shadows with this component. */
	virtual FLightSceneProxy* CreateSceneProxy() const override { return nullptr; }
};

UCLASS(Blueprintable, ClassGroup = (SWRT), meta = (DisplayName = "SWRT Spot Light"))
class PROTOTYPESWRTSHADOWS_API ASWRTSpotLight : public AActor
{
	GENERATED_BODY()

public:
	ASWRTSpotLight();

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Light", meta = (ExposeFunctionCategories = "SpotLight,Rendering|Lighting"))
	TObjectPtr<USWRTSpotLightComponent> SpotLightComponent;

#if WITH_EDITOR
	/**
	 * ASpotLight::EditorApplyScale. The scale gizmo drives the attenuation
	 * radius, Ctrl the outer cone angle and Alt the inner one. An actor that
	 * does not override this is simply scaled, which does nothing to a light
	 * with no mesh.
	 */
	virtual void EditorApplyScale(const FVector& DeltaScale, const FVector* PivotLocation, bool bAltDown, bool bShiftDown, bool bCtrlDown) override;
	virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

#if WITH_EDITORONLY_DATA
	UPROPERTY()
	TObjectPtr<UArrowComponent> ArrowComponent;
#endif
};

/**
 * Derived from FWorldSceneViewExtension rather than FSceneViewExtensionBase:
 * the world flavour already rejects view families belonging to other worlds.
 */
class PROTOTYPESWRTSHADOWS_API FSWRTSpotLightViewExtension : public FWorldSceneViewExtension
{
public:
	FSWRTSpotLightViewExtension(const FAutoRegister& AutoRegister, UWorld* InWorld);

	virtual void SetupViewFamily(FSceneViewFamily& InViewFamily) override {}
	virtual void SetupView(FSceneViewFamily& InViewFamily, FSceneView& InView) override {}
	virtual void BeginRenderViewFamily(FSceneViewFamily& InViewFamily) override;

	virtual void PostRenderBasePassDeferred_RenderThread(
		FRDGBuilder& GraphBuilder,
		FSceneView& InView,
		const FRenderTargetBindingSlots& RenderTargets,
		TRDGUniformBufferRef<FSceneTextureUniformParameters> SceneTextures) override;

protected:
	virtual bool IsActiveThisFrame_Internal(const FSceneViewExtensionContext& Context) const override;

private:
	/**
	 * Written only by a render command enqueued from BeginRenderViewFamily,
	 * read only from PostRenderBasePassDeferred_RenderThread. Never touched
	 * from the game thread, because the render thread can still be working on
	 * the previous frame while the game thread runs ahead.
	 */
	TArray<FSWRTSpotLightRenderData> Lights_RenderThread;
};

/**
 * Owns the view extension. A world subsystem rather than an engine subsystem,
 * because the extension is bound to one UWorld and the light actors it reads
 * live in that world.
 */
UCLASS()
class PROTOTYPESWRTSHADOWS_API USWRTSpotLightSubsystem : public UWorldSubsystem
{
	GENERATED_BODY()

public:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

private:
	TSharedPtr<FSWRTSpotLightViewExtension, ESPMode::ThreadSafe> SceneViewExtension;
};

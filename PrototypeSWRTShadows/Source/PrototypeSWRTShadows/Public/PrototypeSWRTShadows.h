// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

/**
 * Maps the plugin's Shaders folder to the virtual path
 * /Plugins/PrototypeSWRTShadows. Loaded at PostConfigInit so the mapping
 * exists before any shader is compiled.
 */
class FPrototypeSWRTShadowsModule : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};

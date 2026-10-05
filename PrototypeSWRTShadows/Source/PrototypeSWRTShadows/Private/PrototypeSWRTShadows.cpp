// SPDX-License-Identifier: Apache-2.0

#include "PrototypeSWRTShadows.h"

#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "ShaderCore.h"

IMPLEMENT_MODULE(FPrototypeSWRTShadowsModule, PrototypeSWRTShadows);

void FPrototypeSWRTShadowsModule::StartupModule()
{
	UE_LOG(LogTemp, Log, TEXT("PrototypeSWRTShadows: module loaded, compiled %s %s"),
		ANSI_TO_TCHAR(__DATE__), ANSI_TO_TCHAR(__TIME__));

	const TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("PrototypeSWRTShadows"));
	if (Plugin.IsValid())
	{
		AddShaderSourceDirectoryMapping(
			TEXT("/Plugins/PrototypeSWRTShadows"),
			FPaths::Combine(Plugin->GetBaseDir(), TEXT("Shaders")));
	}
}

void FPrototypeSWRTShadowsModule::ShutdownModule()
{
}

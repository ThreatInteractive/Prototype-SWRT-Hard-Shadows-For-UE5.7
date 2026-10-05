// SPDX-License-Identifier: Apache-2.0

using System.IO;
using UnrealBuildTool;

public class PrototypeSWRTShadows : ModuleRules
{
	public PrototypeSWRTShadows(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"CoreUObject",
				"Engine",
				"Projects",
				"RenderCore",
				"Renderer",
				"RHI"
			}
		);

		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"SlateCore",
				"Slate"
			}
		);

		// FViewInfo, FScene and FRDGSystemTextures live in the Renderer's
		// private headers. The light pass needs the view it is rendering and
		// the scene it is culling against, so those paths are required.
		var EngineDir = Path.GetFullPath(Target.RelativeEnginePath);

		PrivateIncludePaths.AddRange(
			new string[]
			{
				Path.Combine(GetModuleDirectory("Renderer"), "Private"),
				Path.Combine(EngineDir, "Source/Runtime/Renderer/Private"),
				Path.Combine(EngineDir, "Source/Runtime/Renderer/Internal")
			}
		);
	}
}

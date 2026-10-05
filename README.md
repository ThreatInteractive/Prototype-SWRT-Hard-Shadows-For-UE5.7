# Prototype SWRT Shadows

A spot light for Unreal Engine 5.7 whose shadows are ray traced in a plain compute shader, with no hardware ray tracing requirement. One hard shadow ray per lit pixel, traced against a two level BVH built from the light's own shadow casters. No shadow map, no screen space trace, nothing carried over from the previous frame. The lighting is the engine's own: the shader includes `DeferredLightingCommon.ush` and calls `GetDynamicLighting`, so only the shadow term differs from an engine spot light.

# Why This Has Been Uploaded & How You Can Help!
It was a considerable amount of work to get the tracer on-par with the hardware DXR path (GA106-300-A1) and since there wasn't any starting material for compute-shader-only raytracing in which modern optimizations/research were compiled into a single repository: it made sense to go ahead and release this project after it hit a quality wall. The problem with hardware-accelerated ray tracing is that its fixed-function architecture is permanently constrained by the research available when the silicon was designed. The graphics programming scene is a constantly evolving space where computational shortcuts and optimizations are constantly being discovered. The vendor agnostic compute-path can implement such optimizations which are not typically compared by leading hardware RT vendors which can sway two imporant scenarios: 

- A consumer's hardware/brand purchase decision when looking at what is actually needed to support current console generation performance-visual standards
- A studio's game development decision on what graphical approach they should spend resources on adopting

  **Here are the topics that need community outsourcing/R&D**

  
- Alpha-testing/masked material support with vertex animation support (important for dynamic hair & foliage assets)
- CPU optimizations
- Soft Shadows (currently relies on MLAA for edge smoothing; temporal denoising & major cost in association are not welcome)
- Better BVH construction with CPU skinning support (it's currently messy since it's working around precompiled UE5 limitations) 
- Vendor/low-Level API optimizations (maybe wave intrinsics, either way support in Vulkan is must)
- 1-frame only ray acceleration research since Threat Interactive's unannounced title deals with frequent camera cuts/fast movement.
- Efficient material branching support for approximated transmission, see [page 264 of this presentation](https://advances.realtimerendering.com/s2023/SIGGRAPH2023-Advances-The-Rendering-of-The-Callisto-Protocol-JimenezPetersen_SlidesOnly.pdf).     


# License
Apache 2.0 (see [LICENSE](LICENSE), and [NOTICE](NOTICE), which has to ship with any copy)

# How To Use
- Enable skin cache in project settings
- Copy this folder to YourProject/Plugins/
- Rebuild the project, since the light is a new actor class
- Place an **SWRT Spot Light** from the Place Actors panel
- It derives from `USpotLightComponent`, so attenuation radius, cone angles, and the transform are the engine's own properties and paste across from an engine spot light for a side by side comparison (there might be a bug with the light intensity value). 

# How A Frame Runs
- Gather, the casters whose shadow could land where the player can see
- Extract, a compute pass writing their triangles into one shared buffer
- Build, a BVH per source on worker threads, a few sources per frame
- Refit, the node bounds on the GPU, every frame, plus a BVH over the placements
- Tile list, per 8x8 tile, which placements its rays can reach
- Trace, one ray per pixel into an R8 mask
- Triangles live in their source's local space, so two placements of one mesh share one copy and one tree, and entering a placement transforms the ray instead of the geometry

# Console Variables
- `r.SWRTSpotLight.Enabled` 1, master switch
- `r.SWRTSpotLight.CasterCull` 1, cull casters against the visible region as well as the cone
- `r.SWRTSpotLight.CasterLog` 0, 1 counts per light, 2 names every primitive and why it was kept
- `r.SWRTSpotLight.Debug` 0, 1 shadow term, 2 lane waste, 3 node visits per ray, 4 warp mixed
- `r.SWRTSpotLight.Trace.HeatmapRange` 42, node visits that paint full red in view 3
- any nonzero Debug also compiles the per thread counters in and writes `Saved/PrototypeSWRTShadows/SWRTTraceLatest.log`, so nothing timed should be measured with it on

# Author
Kevin Jimenez

# Thanks To
* Tomas Möller and Ben Trumbore (ray triangle intersection with no plane equation stored; https://www.graphics.cornell.edu/pubs/1997/MT97.html)
* Jeffrey Goldsmith and John Salmon (the surface area heuristic; https://doi.org/10.1109/MCG.1987.276983)
* J.D. MacDonald and K.S. Booth (the surface area heuristic, and where the split belongs; https://link.springer.com/article/10.1007/BF01911006)
* Ingo Wald (binned surface area construction, cheap enough to rebuild; https://www.sci.utah.edu/~wald/Publications/2007/ParallelBVHBuild/fastbuild.pdf)
* Henri Ylitie, Tero Karras and Samuli Laine (four wide nodes with bounds quantised to 16 bits an axis; https://research.nvidia.com/publication/2017-07_efficient-incoherent-ray-traversal-gpus-through-compressed-wide-bvhs)
* Ingo Wald, Philipp Slusallek, Carsten Benthin and Markus Wagner (a packet of rays tested against a box once; https://www.sci.utah.edu/~wald/Publications/2001/CRT/CRT.pdf)
* Louis Bavoil (thread group ID swizzling for L2 locality; https://developer.nvidia.com/blog/optimizing-compute-shaders-for-l2-locality-using-thread-group-id-swizzling/)
* Kostas Anagnostou (surface area trees, child ordering and typed float4 buffers, each measured in his hybrid shadow series; https://interplayoflight.wordpress.com/2018/09/04/hybrid-raytraced-shadows-part-2-performance-improvements/)
* Yuriy O'Donnell (widest child first, the same ordering reached from the other direction; https://github.com/kayru/RayTracedShadows)
* The Forge Interactive (a triangle as one vertex and two edges, the instance root box test, ordered child traversal, from 09a_HybridRaytracing in tag v1.45; https://github.com/ConfettiFX/The-Forge)
* Jacco Bikker for his research in BVH construction: https://github.com/jbikker/tinybvh
* Sebastian Aaltonen, Timothy Lottes, and Kostas Anagnostou for their several articles/blogs/twitter(X) post about hardware utilization.
* Every Threat Interactive [YouTube subscriber](https://www.youtube.com/@ThreatInteractive) & Patreon member: [join here to support our team](https://www.patreon.com/c/THREATINTERACTIVE).    

# AI-Use Notice
AI was mostly used organize custom performance data & ensure full compatibility of human graphic-derived-concepts with UE's plugin-limited API. It was simply used to assist in this projects development. AI did not "create" or in other words: steal from existing non-open source projects that already exist as everything in this project went through constant manual verification methods to ensure all approaches are fully traceable to internal research notes, Creative Commons academic papers, and compliant open-source repositories. If something seems unclear/redundant in the code; feel free to reach out and suggest your own improvement as that's the whole reason this is being uploaded.

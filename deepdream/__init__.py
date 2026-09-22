"""Runtime package for the TouchDeepDream component.

TouchDesigner imports the submodules it needs. Importing this package does
not load PyTorch, so the cook thread does not create a CUDA context.
"""

# README

## Build

The applications build is completely defined in `CMakeApplicationPresets.json`.
To build this application using CMake Presets you simply need to include the path to `CMakeApplicationPresets.json` in our `CMakeUserPresets.json` file.
After that you can build the application with:

```bash
cmake --workflow --preset MTWBenchmark
```

### CMakeUserPresets

Create a file named `CMakeUserPresets.json` in your projects root folder.
The file must contain at least:

```json
{
  "version": 9,
  "cmakeMinimumRequired": {
    "major": 3,
    "minor": 25,
    "patch": 0
  },
  "include": [
    "${sourceDir}/apps/benchmarks/MaterialTransport/particles_total_enthalpy_cht/hpc-performance/CMakeApplicationPresets.json"
  ],
  "configurePresets": [],
  "buildPresets": [],
  "testPresets": [],
  "workflowPresets": []
}
```

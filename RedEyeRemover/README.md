# Remove Red Eye with OpenCV and C++

## How to run

1. __Step 1:__ create build directory

```
mkdir build
cd build
```

2. __Step 2:__ create VS project files using cmake

```
cmake - G "Visual Studio 16 2019" ..
```

3. __Step 3:__ build our application

```
cmake --build . --config Release
```

4. __Step 4:__ run our application

```
cd .. 
.\build\Release\removeRedEyes.exe
```

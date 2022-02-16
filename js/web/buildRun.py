import os
import shutil

Model_Path = ".\\test\\data\\f\\"
Build_Path = "..\\..\\build\\Windows\\Debug\\"
Dist_Path = ".\\dist\\"
Binding_Path = ".\\lib\\wasm\\binding\\"

## Building WASM - Release - WASM_SIMD
command = "..\\..\\build.bat --build_wasm --skip_tests --skip_submodule_sync --config Debug --enable_wasm_simd --emsdk_version releases-upstream-823d37b15d1ab61bc9ac0665ceef6951d3703842-64bit"
os.system(command)

## Coping files
file_name = "ort-wasm-simd.wasm"
shutil.copyfile(Build_Path+file_name,Dist_Path+file_name)

file_name = "ort-wasm.js"
shutil.copyfile(Build_Path+file_name,Binding_Path+file_name)

file_name = "ort-wasm-threaded.js"
shutil.copyfile(Build_Path+file_name,Binding_Path+file_name)

file_name = "ort-wasm-threaded.worker.js"
shutil.copyfile(Build_Path+file_name,Binding_Path+file_name)

## Running the model
command = "npm test -- model "+Model_Path+" -b=wasm --wasm-number-threads=1"

os.system(command)
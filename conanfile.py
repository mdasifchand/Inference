from conans import ConanFile, CMake

class InferenceTensorRT(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    requires = "opencv/4.5.1"
    generators = "cmake_find_package"


    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()


#conan install .. -c tools.system.package_manager:mode=install -c tools.system.package_manager:sudo=True 
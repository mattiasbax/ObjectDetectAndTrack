#include <gtest/gtest.h>
#include <ObjectDetectorHandler.h>
#include <string>

TEST(MyClass, defaultConstruction) {
    std::string yoloName = "yolo3";
    ObjectDetectorHandler odh(std::move(yoloName));
    ASSERT_TRUE(true);
}

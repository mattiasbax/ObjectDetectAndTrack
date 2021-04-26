#include "gtest/gtest.h"
#include "../../src/MyClass.h"

TEST(MyClass, defaultConstruction) {
    constexpr int initValue = 5;
    const MyClass myClass(initValue);
    EXPECT_EQ(myClass.getMyInt(), initValue);
}

TEST(MyClass, setValue) {
    constexpr int initValue = 0;
    constexpr int setValue = 1;
    MyClass myClass(initValue);
    myClass.setMyInt(setValue);
    EXPECT_EQ(myClass.getMyInt(), setValue);
}
#include <iostream>
#include "MyClass.h"

int main() {
    MyClass myClass(8);
    std::cout << "Hello, World: " << myClass.getMyInt() << std::endl;
    return 0;
}

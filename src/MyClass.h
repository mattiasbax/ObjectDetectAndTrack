#pragma once

class MyClass
{
public:
    MyClass(const int myInt) : MyInt(myInt) {};
    int getMyInt() const;
    void setMyInt(const int myInt);
private:
    int MyInt;
};
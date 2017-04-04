#include<iostream>
#include<string>
using namespace std;

void PrintManyTimes (const string& input, int times) {
    for(int i = 0; i < times; ++i) {
        cout << input << endl;
    }
};

int main() {
    string out = "Hello World";
    PrintManyTimes(out, 3);
}

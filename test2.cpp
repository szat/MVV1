#include<iostream>
#include<stdio.h>
#include<array>
using namespace std;

void GhostFct() {
    cout << "this fct does not do anything" << endl;
}

int printer(){
    printf("tester");
    return 0;
}


int main() {
    for(int i = 0; i < 10; ++i) {
        for(int j = 0; j < 10; ++j) {
            if( i+j %2 == 0) cout << "idk what this is doing, but i is " << i << " and j is " << j << endl;
            //else cout << "i + j was not even" << endl;
        }
    }

    //A few ghost functions
    GhostFct();
    GhostFct();
    GhostFct();

    int i = 0;
    printer();
}


// extra comments at the end

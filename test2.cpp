<<<<<<< HEAD
#include<iostream>
using namespace std;

void GhostFct() {
    cout << "this fct does not do anything" << endl;
}

int main() {
    for(int i = 0; i < 10; ++i) {
        for(int j = 0; j < 10; ++j) {
            if( i+j %2 == 0) cout << "idk what this is doing, but i is " << i << " and j is " << j << endl;
            else cout << "i + j was not even" << endl;
        }
    }
    return 0;
}
=======
#include<stdio.h>
#include<array>

int printer(){
    printf("tester");
    return 0;
}

// more comments

int main(){
    //testing

    int i = 0;
    printer();
    //testing some more
    //lots of comments here
}


// extra comments at the end
>>>>>>> be3b5ce66f184e5eaae6d622e948a096e85d8fe8

#include <stdio.h>

bool TestPrime(int number){
    // This is really not a test for primes
    // I am just trying out C++


    if (number == 2 || number == 3 || number == 5)
        return false;
    if (number % 2 == 0)
        return false;
    else if (number % 3 == 0)
        return false;
    else if (number % 5 == 0)
        return false;
    else
        return true;
}

int main()
{
    printf("Testing primes");

    for (int i = 0; i < 100; i++){
        if (TestPrime(i)){
            printf("%d\n", i);
        }
    }

    return 0;
}





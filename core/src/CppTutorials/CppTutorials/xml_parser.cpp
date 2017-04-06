#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

class Tag {
    protected:
        vector<Tag> subtags;
        string name;
        string value;
    public:
        Tag(string inputName, string inputValue){
            name = inputName;
            value = inputValue;
        }
        vector<Tag> get_subtags(){
            return subtags;
        }
        void set_name(string input){
            name = input;
        }
        string get_name(){
            if (name == "")
                return "Not Found!";
            return name;
        }
        void set_value(string input){
            value = input;
        }
        string get_value(){
            if (value == "")
                return "Not Found!";
            return value;
        }
};

void print_string_vector(vector<string> input){
    int len = input.size();
    for (int i = 0; i < len; i++){
        cout << input[i] << '\n';
    }
}

vector<string> input_to_string_vectors(vector<string> input, int numInputs){
    string str;
    for (int i = 0; i < numInputs; i++){
        getline(cin, str);
        if (str == "\n" || str == "\r" || str == "\r\n"){
            i--;    
        } else {
            input.push_back(str);
        }
    }
    return input;

}

vector<string> remove_angle_brackets(vector<string> input){
    int numTags = input.size();
    for (int i = 0; i < numTags; i++){
        int len = input[i].size();
        input[i] = input[i].substr(1,len-2);
    }
    return input;
}

vector<string> recursive_split(){
    vector<string> result = {};
    return result;
}

int main() {
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */   
    
    string str;
    getline(cin, str);
    
    int numTags, numRequests;
    numTags = 4;
    numRequests = 3;
    vector<string> tags = {};
    tags = input_to_string_vectors(tags, numTags);
    vector<string> requests = {};
    requests = input_to_string_vectors(requests, numRequests);
    
    // Ok great, we have the tags and the vectors.
    tags = remove_angle_brackets(tags);
    
    print_string_vector(tags);
    print_string_vector(requests);
    
    
    
    Tag *tag1 = new Tag("", "HelloWorld");
    Tag *tag2 = new Tag("Name1", "");
    
    
    //cout << (*tag1).get_subtags()[0].get_name();
    //cout << (*tag1).get_name();
    //cout << (*tag1).get_value();
    
    return 0;
}

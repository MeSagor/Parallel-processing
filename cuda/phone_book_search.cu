
#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda_runtime.h>

using namespace std;

const int MAX_CONTACTS = 8000; // Maximum number of contacts
const int MAX_NAME_LENGTH = 60;
const int MAX_PHONE_LENGTH = 20;
const int BLOCK_SIZE = 256;


// Define a const char array that can be used on both host and device
__constant__ char targetName[MAX_NAME_LENGTH];

// CUDA kernel for searching contacts matching a given name
// CUDA kernel for searching contacts matching a given name
__global__ void searchContactsKernel(char* names, int numContacts, int* matchingIndices, int contactsPerThread) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int startIdx = tid * contactsPerThread;
    int endIdx = startIdx + contactsPerThread;

    for (int i = startIdx; i < endIdx && i < numContacts; ++i) {
        // Custom implementation of string comparison
        int j = 0;
        while (targetName[j] != '\0' && names[i * MAX_NAME_LENGTH + j] == targetName[j]) {
            ++j;
        }

        if (targetName[j] == '\0' && names[i * MAX_NAME_LENGTH + j] == '\0') {
            matchingIndices[i] = i;
        } else {
            matchingIndices[i] = -1;
        }
    }
}



// Function to read phonebook data from files
int readPhonebookData(const string& filename, char* names, char* phoneNumbers) {
    ifstream file(filename);
    int numContacts = 0;

    if (file.is_open()) {
        string line;
        while (getline(file, line) && numContacts < MAX_CONTACTS) {
            size_t spacePos = line.find(' ');
            if (spacePos != string::npos) {
                strncpy(names + numContacts * MAX_NAME_LENGTH, line.substr(0, spacePos).c_str(), MAX_NAME_LENGTH - 1);
                names[(numContacts + 1) * MAX_NAME_LENGTH - 1] = '\0';
                strncpy(phoneNumbers + numContacts * MAX_PHONE_LENGTH, line.substr(spacePos + 1).c_str(), MAX_PHONE_LENGTH - 1);
                phoneNumbers[(numContacts + 1) * MAX_PHONE_LENGTH - 1] = '\0';
                ++numContacts;
            }
        }
        file.close();
    }
    return numContacts;
}

int main(int argc, char* argv[]) {
    //if (argc < 2) {
    //    cout << "Usage: " << argv[0] << " <file1> <file2> <file3> ..." << endl;
    //    return 1;
    //}

   // const int numFiles = argc - 1;

    const string filenames[] = { "input.txt" };



    // const char targetName[] = "Webster__Daniel";
    const char hostTargetName[MAX_NAME_LENGTH] = "Vela__Filemon";

    char* allNames = new char[MAX_CONTACTS * MAX_NAME_LENGTH];
    char* allPhoneNumbers = new char[MAX_CONTACTS * MAX_PHONE_LENGTH];
    int totalContacts = 0;

    int numFiles = sizeof(filenames) / sizeof(filenames[0]);

    for (int i = 0; i < numFiles; ++i) {
        int contactsInFile = readPhonebookData(filenames[i], allNames + totalContacts * MAX_NAME_LENGTH, allPhoneNumbers + totalContacts * MAX_PHONE_LENGTH);
        totalContacts += contactsInFile;
    }

    cout << totalContacts << endl;

    char* d_allNames;
    int* d_matchingIndices;
    cudaMalloc((void**)&d_allNames, totalContacts * MAX_NAME_LENGTH);
    cudaMalloc((void**)&d_matchingIndices, totalContacts * sizeof(int));

    cudaMemcpy(d_allNames, allNames, totalContacts * MAX_NAME_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(targetName, hostTargetName, MAX_NAME_LENGTH * sizeof(char));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    // Launch CUDA kernel
    // int numBlocks = (totalContacts + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // searchContactsKernel<<<numBlocks, BLOCK_SIZE>>>(d_allNames, totalContacts, d_matchingIndices);
    int numThreads = 1;
    int contactsPerThread = totalContacts / numThreads;
    searchContactsKernel << <1, numThreads >> > (d_allNames, totalContacts, d_matchingIndices, contactsPerThread);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    int* h_matchingIndices = new int[totalContacts];
    cudaMemcpy(h_matchingIndices, d_matchingIndices, totalContacts * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the matching contacts
    cout << "total Contacts: " << totalContacts << endl;
    cout << "Matching Contacts for '" << hostTargetName << "':" << endl;
    for (int i = 0; i < totalContacts; ++i) {
        if (h_matchingIndices[i] != -1) {
            cout << allNames + h_matchingIndices[i] * MAX_NAME_LENGTH << "\t" << allPhoneNumbers + h_matchingIndices[i] * MAX_PHONE_LENGTH << endl;
        }
    }

    float mili;
    cudaEventElapsedTime(&mili, start, end);
    cout << mili << " time passed" << endl;

    // Free allocated memory
    delete[] allNames;
    delete[] allPhoneNumbers;
    delete[] h_matchingIndices;
    cudaFree(d_allNames);
    cudaFree(d_matchingIndices);

    return 0;
}

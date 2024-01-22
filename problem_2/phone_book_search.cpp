#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <mpi.h>

using namespace std;

const int MAX_CONTACTS = 8000; // Maximum number of contacts
const int MAX_NAME_LENGTH = 60;
const int MAX_PHONE_LENGTH = 20;

// Function to read phonebook data from files
int readPhonebookData(const char* filename, char* names, char* phoneNumbers) {
    ifstream file(filename);
    int numContacts = 0;
    // cout<<filename<<endl;
    if (file.is_open()) {
        // cout<<"file is open"<<endl;
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

// Function to search for contacts matching a given name
vector<int> searchContacts(char* names, int numContacts, const string& targetName) {
    vector<int> matchingIndices;
    for (int i = 0; i < numContacts; ++i) {
        string singleName(names + i * MAX_NAME_LENGTH, strlen(names + i * MAX_NAME_LENGTH));
        for (char& c : singleName) c = tolower(c);
        // cout << singleName << ' ' << singleName.length() << endl;
        if (strcmp(singleName.c_str(), targetName.c_str()) == 0) {
            matchingIndices.push_back(i);
        }
    }
    return matchingIndices;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    if (argc < 2) {
        if (rank == 0) {
            cout << "Usage: " << argv[0] << " <file1> <file2> <file3> ..." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    const char filenames[][60] = { "/Users/SAGOR/Unknown/PP_Lab/problem_2/input.txt",
    "/Users/SAGOR/Unknown/PP_Lab/problem_2/input2.txt",
    "/Users/SAGOR/Unknown/PP_Lab/problem_2/input3.txt",
    "/Users/SAGOR/Unknown/PP_Lab/problem_2/input4.txt" };

    char allNames[MAX_CONTACTS * MAX_NAME_LENGTH];
    char allPhoneNumbers[MAX_CONTACTS * MAX_PHONE_LENGTH];
    int totalContacts = 0;

    double start = MPI_Wtime();

    int numFiles = sizeof(filenames) / sizeof(filenames[0]);
    if (rank == 0) {
        // Read phonebook data from files in the master process (rank 0)
        int contactsPerFile[argc - 1];
        for (int i = 1; i < argc; ++i) {
            contactsPerFile[i - 1] = readPhonebookData(argv[i], allNames + totalContacts * MAX_NAME_LENGTH, allPhoneNumbers + totalContacts * MAX_PHONE_LENGTH);
            totalContacts += contactsPerFile[i - 1];
        }
    }



    // cout << "total Contacts: " << totalContacts << endl;

    // Broadcast the total number of contacts to all processes
    MPI_Bcast(&totalContacts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Distribute the contacts among processes
    int contactsPerProcess = totalContacts / numProcesses;
    char localNames[MAX_CONTACTS * MAX_NAME_LENGTH];
    char localPhoneNumbers[MAX_CONTACTS * MAX_PHONE_LENGTH];
    MPI_Scatter(allNames, contactsPerProcess * MAX_NAME_LENGTH, MPI_CHAR,
        localNames, contactsPerProcess * MAX_NAME_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);

    MPI_Scatter(allPhoneNumbers, contactsPerProcess * MAX_PHONE_LENGTH, MPI_CHAR,
        localPhoneNumbers, contactsPerProcess * MAX_PHONE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Search for contacts with a specific name (change "John Doe" to your desired name)
    string targetName = "WEBster__Daniel", targetNameLower="";
    // for (int i = 0; targetName[i] != '\0'; i++) {
    //     // targetName[i] = tolower(targetName[i]);
    //     cout<<targetName[i]<<endl;
    // }
    for (char& c : targetName) targetNameLower += tolower(c);
    vector<int> matchingIndices = searchContacts(localNames, contactsPerProcess, targetNameLower);
    vector<int> matchingIndices2(contactsPerProcess, -1);
    for (int x : matchingIndices) {
        matchingIndices2[x] = x;
    }

    // Gather the matching contacts indices to the master process
    vector<int> allMatchingIndices(totalContacts);
    MPI_Gather(&matchingIndices2[0], contactsPerProcess, MPI_INT,
        &allMatchingIndices[0], contactsPerProcess, MPI_INT, 0, MPI_COMM_WORLD);


    // MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0) {
        // Print the matching contacts in the master process
        cout << "total Contacts: " << totalContacts << endl;
        cout << "Matching Contacts for '" << targetName << "':" << endl;
        for (const auto& index : allMatchingIndices) {
            if (index == -1) continue;
            cout << allNames + index * MAX_NAME_LENGTH << "\t" << allPhoneNumbers + index * MAX_PHONE_LENGTH << endl;
        }
        cout << "Time taken: " << end - start << endl;
    }

    MPI_Finalize();

    return 0;
}

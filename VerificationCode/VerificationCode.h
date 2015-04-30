// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the VERIFICATIONCODE_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// VERIFICATIONCODE_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef VERIFICATIONCODE_EXPORTS
#define VERIFICATIONCODE_API __declspec(dllexport)
#else
#define VERIFICATIONCODE_API __declspec(dllimport)
#endif


	
extern "C" VERIFICATIONCODE_API int InitEngine();

extern "C" VERIFICATIONCODE_API int RecognizeCode(char* filePath, char* code, float* conf);

extern "C" VERIFICATIONCODE_API int ReleaseEngine();

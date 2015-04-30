#include "stdafx.h"


int get_image_list(string path, vector<string>& image_vec)
{
	intptr_t stHandle;
	struct _finddata_t stFileinfo;
	string strFilespec = path + "\\*.*";

	if ((stHandle = _findfirst(strFilespec.c_str(), &stFileinfo)) != -1L)
	{
		do 
		{	
			if (stFileinfo.name[0] == '.')
				continue;

			if (_A_SUBDIR & stFileinfo.attrib)
				continue;

			char ext[8];
			char fileName[256];
			_splitpath(stFileinfo.name, NULL, NULL, fileName, ext);

			if ( string(_strupr(ext)) == string(".BMP") || string(_strupr(ext)) == string(".JPG") )
				image_vec.push_back(string(stFileinfo.name));
		}
		while(_findnext(stHandle, &stFileinfo) == 0);
	}

	return 0;
}

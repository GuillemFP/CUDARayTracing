#ifndef CONFIG_H
#define CONFIG_H

typedef struct json_object_t JSON_Object;
typedef struct json_array_t  JSON_Array;
typedef struct json_value_t  JSON_Value;

class ConfigArray;

class Config
{
public:
	Config(const char* file);
	Config(JSON_Object* section);
	~Config();

	bool IsLoaded() const;
	bool HasSection(const char* name, int index = -1) const;
	bool HasArray(const char* name, int index = -1) const;
	Config GetSection(const char* name, int index = -1) const;
	ConfigArray GetArray(const char* name, int index = -1) const;

	const char* GetString(const char* name, const char* default_value = nullptr, int index = -1) const;
	const char* GetStringRequired(const char* name, int index = -1) const;
	bool GetBool(const char* name, bool default_value = false, int index = -1) const;
	bool GetBoolRequired(const char* name, int index = -1) const;
	int GetInt(const char* name, int default_value = 0, int index = -1) const;
	int GetIntRequired(const char* name, int index = -1) const;
	float GetFloat(const char* name, float default_value = 0, int index = -1) const;
	float GetFloatRequired(const char* name, int index = -1) const;

	size_t GetArrayLength(const char* name) const;

protected:
	JSON_Value* FindValue(const char* name, int index = -1) const;
	JSON_Array* FindArray(const char* name, int index = -1) const;
	JSON_Object* FindObject(const char* name, int index = 1) const;

private:
	JSON_Object* root = nullptr;
};

class ConfigArray
{
public:
	ConfigArray(const char* file);
	ConfigArray(JSON_Array* array);
	~ConfigArray();

	bool IsLoaded() const;
	bool HasSection(int index, const char* name = nullptr) const;
	bool HasArray(int index, const char* name = nullptr) const;
	Config GetSection(int index, const char* name = nullptr) const;
	ConfigArray GetArray(int index, const char* name = nullptr) const;

	const char* GetString(int index, const char* default_value = nullptr, const char* name = nullptr) const;
	const char* GetStringRequired(int index, const char* name = nullptr) const;
	bool GetBool(int index, bool default_value = false, const char* name = nullptr) const;
	bool GetBoolRequired(int index, const char* name = nullptr) const;
	int GetInt(int index, int default_value = 0, const char* name = nullptr) const;
	int GetIntRequired(int index, const char* name = nullptr) const;
	float GetFloat(int index, float default_value = 0, const char* name = nullptr) const;
	float GetFloatRequired(int index, const char* name = nullptr) const;

	size_t GetArrayLength() const;

protected:
	JSON_Value* FindValue(int index, const char* name = nullptr) const;
	JSON_Array* FindArray(int index, const char* name = nullptr) const;
	JSON_Object* FindObject(int index, const char* name = nullptr) const;

private:
	JSON_Array* root = nullptr;
};

#endif // !CONFIG_H
#include "Config.h"

#include "parson\parson.h"
#include <assert.h>

Config::Config(const char* file)
{
	root = json_value_get_object(json_parse_file(file));
}

Config::Config(JSON_Object* section) : root(section)
{
}

Config::~Config()
{
}

bool Config::IsLoaded() const
{
	return root != nullptr;
}

bool Config::HasSection(const char* name, int index) const
{
	return FindObject(name, index);
}

bool Config::HasArray(const char * name, int index) const
{
	return FindArray(name, index);
}

Config Config::GetSection(const char* name, int index) const
{
	return Config(FindObject(name, index));
}

ConfigArray Config::GetArray(const char * name, int index) const
{
	return ConfigArray(FindArray(name, index));
}

const char* Config::GetString(const char* name, const char* default_value, int index) const
{
	JSON_Value* value = FindValue(name, index);

	if (value && json_value_get_type(value) == JSONString)
		return json_value_get_string(value);

	return default_value;
}

const char* Config::GetStringRequired(const char* name, int index) const
{
	JSON_Value* value = FindValue(name, index);

	assert(value && json_value_get_type(value) == JSONString);
	return json_value_get_string(value);
}

bool Config::GetBool(const char* name, bool default_value, int index) const
{
	JSON_Value* value = FindValue(name, index);

	if (value && json_value_get_type(value) == JSONBoolean)
		return json_value_get_boolean(value) != 0;

	return default_value;
}

bool Config::GetBoolRequired(const char* name, int index) const
{
	JSON_Value* value = FindValue(name, index);

	assert(value && json_value_get_type(value) == JSONBoolean);
	return json_value_get_boolean(value) != 0;
}

int Config::GetInt(const char* name, int default_value, int index) const
{
	JSON_Value* value = FindValue(name, index);

	if (value && json_value_get_type(value) == JSONNumber)
		return (int)json_value_get_number(value);

	return default_value;
}

int Config::GetIntRequired(const char* name, int index) const
{
	JSON_Value* value = FindValue(name, index);

	assert(value && json_value_get_type(value) == JSONNumber);
	return (int)json_value_get_number(value);
}

float Config::GetFloat(const char* name, float default_value, int index) const
{
	JSON_Value* value = FindValue(name, index);

	if (value && json_value_get_type(value) == JSONNumber)
		return (float)json_value_get_number(value);

	return default_value;
}

float Config::GetFloatRequired(const char* name, int index) const
{
	JSON_Value* value = FindValue(name, index);

	assert(value && json_value_get_type(value) == JSONNumber);
	return (float)json_value_get_number(value);
}

size_t Config::GetArrayLength(const char * name) const
{
	JSON_Array* array = json_object_get_array(root, name);
	if (array)
		return json_array_get_count(array);
	return 0;
}

JSON_Value* Config::FindValue(const char* name, int index) const
{
	if (index < 0)
		return json_object_get_value(root, name);

	JSON_Array* value_array = json_object_get_array(root, name);
	if (value_array)
		return json_array_get_value(value_array, index);

	return nullptr;
}

JSON_Array* Config::FindArray(const char* name, int index) const
{
	if (index < 0)
		return json_object_get_array(root, name);

	JSON_Array* array_array = json_object_get_array(root, name);
	if (array_array)
		return json_array_get_array(array_array, index);

	return nullptr;
}

JSON_Object* Config::FindObject(const char* name, int index) const
{
	if (index < 0)
		return json_object_get_object(root, name);

	JSON_Array* object_array = json_object_get_array(root, name);
	if (object_array)
		return json_array_get_object(object_array, index);

	return nullptr;
}

ConfigArray::ConfigArray(const char* file)
{
	root = json_value_get_array(json_parse_file(file));
}

ConfigArray::ConfigArray(JSON_Array* array) : root(array)
{
}

ConfigArray::~ConfigArray()
{
}

bool ConfigArray::IsLoaded() const
{
	return root != nullptr;
}

bool ConfigArray::HasSection(int index, const char * name) const
{
	return FindObject(index, name);
}

bool ConfigArray::HasArray(int index, const char * name) const
{
	return FindArray(index, name);
}

Config ConfigArray::GetSection(int index, const char* name) const
{
	return Config(FindObject(index, name));
}

ConfigArray ConfigArray::GetArray(int index, const char* name) const
{
	return ConfigArray(FindArray(index, name));
}

const char* ConfigArray::GetString(int index, const char* default_value, const char* name) const
{
	JSON_Value* value = FindValue(index, name);

	if (value && json_value_get_type(value) == JSONString)
		return json_value_get_string(value);

	return default_value;
}

const char * ConfigArray::GetStringRequired(int index, const char* name) const
{
	JSON_Value* value = FindValue(index, name);

	assert(value && json_value_get_type(value) == JSONString);
	return json_value_get_string(value);
}

bool ConfigArray::GetBool(int index, bool default_value, const char* name) const
{
	JSON_Value* value = FindValue(index, name);

	if (value && json_value_get_type(value) == JSONBoolean)
		return json_value_get_boolean(value) != 0;

	return default_value;
}

bool ConfigArray::GetBoolRequired(int index, const char* name) const
{
	JSON_Value* value = FindValue(index, name);

	assert(value && json_value_get_type(value) == JSONBoolean);
	return json_value_get_boolean(value) != 0;
}

int ConfigArray::GetInt(int index, int default_value, const char* name) const
{
	JSON_Value* value = FindValue(index, name);

	if (value && json_value_get_type(value) == JSONNumber)
		return (int)json_value_get_number(value);

	return default_value;
}

int ConfigArray::GetIntRequired(int index, const char* name) const
{
	JSON_Value* value = FindValue(index, name);

	assert(value && json_value_get_type(value) == JSONNumber);
	return (int)json_value_get_number(value);
}

float ConfigArray::GetFloat(int index, float default_value, const char* name) const
{
	JSON_Value* value = FindValue(index, name);

	if (value && json_value_get_type(value) == JSONNumber)
		return (float)json_value_get_number(value);

	return default_value;
}

float ConfigArray::GetFloatRequired(int index, const char* name) const
{
	JSON_Value* value = FindValue(index, name);

	assert(value && json_value_get_type(value) == JSONNumber);
	return (float)json_value_get_number(value);
}

size_t ConfigArray::GetArrayLength() const
{
	return json_array_get_count(root);
}

JSON_Value* ConfigArray::FindValue(int index, const char* name) const
{
	if (!name)
		return json_array_get_value(root, index);

	JSON_Object* object = json_array_get_object(root, index);
	if (object)
		return json_object_get_value(object, name);

	return nullptr;
}

JSON_Array* ConfigArray::FindArray(int index, const char* name) const
{
	if (!name)
		return json_array_get_array(root, index);

	JSON_Object* object = json_array_get_object(root, index);
	if (object)
		return json_object_get_array(object, name);

	return nullptr;
}

JSON_Object* ConfigArray::FindObject(int index, const char* name) const
{
	if (!name)
		return json_array_get_object(root, index);

	JSON_Object* object = json_array_get_object(root, index);
	if (object)
		return json_object_get_object(object, name);

	return nullptr;
}

#ifndef ENTITYLIST_H
#define ENTITYLIST_H

#include "Entity.h"

class EntityList
{
public:
	__device__ EntityList(int listSize) : _listSize(listSize) 
	{
		_list = new Entity*[_listSize];
	}

	__device__ ~EntityList()
	{
		for (int i = 0; i < _numEntities; i++)
		{
			delete _list[i];
		}

		delete[] _list;
	}

	__device__ bool push_back(Entity* entity)
	{
		if (_numEntities == _listSize)
		{
			return false;
		}

		_list[_numEntities] = entity;
		++_numEntities;
	}

	__device__ bool Hit(const Ray& ray, float minDist, float maxDist, HitInfo& hitInfo) const
	{
		HitInfo currentHitInfo;
		float currentMaxDistance = maxDist;
		for (int i = 0; i < _numEntities; i++)
		{
			if (_list[i]->Hit(ray, minDist, currentMaxDistance, currentHitInfo))
			{
				currentMaxDistance = currentHitInfo.distance;
				hitInfo = currentHitInfo;
			}
		}

		return hitInfo.isHit;
	}

private:
	Entity** _list;
	int _listSize = 0;
	int _numEntities = 0;
};

#endif // !ENTITYLIST_H

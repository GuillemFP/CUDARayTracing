#ifndef ENTITYLIST_H
#define ENTITYLIST_H

#include "Entity.h"

class EntityList : public Entity
{
public:
	__device__ EntityList() = default;
	__device__ EntityList(Entity** list, int listSize) : _list(list), _listSize(listSize) {}
	__device__ bool Hit(const Ray& ray, float minDist, float maxDist, HitInfo& hitInfo) const
	{
		HitInfo currentHitInfo;
		float currentMaxDistance = maxDist;
		for (int i = 0; i < _listSize; i++)
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
	int _listSize;
};

#endif // !ENTITYLIST_H

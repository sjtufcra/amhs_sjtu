import asyncio
import redis.asyncio as rds
from aiocache import Cache
from aiocache.serializers import JsonSerializer


redis = rds.from_url('redis://10.34.58.42:6379',decode_responses=True)
cache = Cache(Cache.MEMORY,serializer=JsonSerializer())

async def cache_redis():
  keys = await redis.keys(pattern='Car:monitor:*')
  values = []
  for key in keys:
   value = await redis.get(key)
   values.append(value)
  return values
  

async def cache_date():
  data = await cache_redis()
  await cache.set('care_data',data)
  print('data:cache')

async def main():
  count = 0
  while True:
    asyncio.create_task(cache_date())
    print(f'run:{count}')
    car_date = await cache.get('care_data')
    if car_date:
      count +=1
      data = len(car_date)
      print('car is :',data)
    else:
      print('no')

asyncio.run(main())

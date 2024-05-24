from pymilvus import *

client = MilvusClient(
    uri="http://192.168.1.111:19530"
)
collection = 'image'


def load_collection():
    client.load_collection(
        collection_name=collection
    )

def list_partitions():
    res = client.list_partitions(collection_name=collection)
    print(res)

def load_partitions(partitions: list):
    client.load_partitions(
        collection_name=collection,
        partition_names=partitions
    )

def create_partition(partition):
    client.create_partition(
        collection_name=collection,
        partition_name=partition
    )

def release_partitions(partitions: list):
    client.release_partitions(
        collection_name=collection,
        partition_names=partitions
    )

# Before dropping a partition, you need to release it from memory.
def drop_partition(partition:list):
    release_partitions(partition)
    client.drop_partitions(
        collection_name=collection,
        partition_names=partition
    )

def get_sizeof_partition(partition:list):
    res = client.query(
        collection_name=collection,
        filter="",
        partition_names=partition,
        output_fields=["count(*)"]
    )
    size = res[0]["count(*)"]
    return size

def load_data_from_partition(partition):
    load_partitions([partition])
    res = client.query(
        collection_name=collection,
        filter="",
        partition_names=[partition],
        limit=16384                 # max limit of Milvus
    )
    release_partitions([partition])
    return res                      # dict


def insert_data_to_partition(partition, data):
    '''
    #####   Schema of collection    #####
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=512)
    schema.add_field(field_name="image_url", datatype=DataType.VARCHAR, max_length=200)
    schema.add_field(field_name="image_description", datatype=DataType.VARCHAR, max_length=10000)
    schema.add_field(field_name="ai_description", datatype=DataType.VARCHAR, max_length=10000)
    schema.add_field(field_name="keywords", datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=4096, max_length=1000)
    schema.add_field(field_name="group", datatype=DataType.INT64)
    schema.add_field(field_name="subgroup", datatype=DataType.INT64)
    '''
    load_partitions([partition])
    res = client.insert(
        collection_name=collection,
        data=data,
        partition_name=partition
    )
    release_partitions([partition])

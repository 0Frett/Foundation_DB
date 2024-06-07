from pymilvus import *

client = MilvusClient(
    uri="http://192.168.1.111:19530"
)

# create db
schema = MilvusClient.create_schema(
    auto_id=True,
    enable_dynamic_field=True,
)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=512)
schema.add_field(field_name="image_url", datatype=DataType.VARCHAR, max_length=200)
schema.add_field(field_name="image_description", datatype=DataType.VARCHAR, max_length=10000)
schema.add_field(field_name="ai_description", datatype=DataType.VARCHAR, max_length=10000)
schema.add_field(field_name="keywords", datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=4096, max_length=1000)

index_params = client.prepare_index_params()

index_params.add_index(
    field_name="id",
    index_type="STL_SORT"
)

index_params.add_index(
    field_name="vector", 
    index_type="IVF_FLAT",
    metric_type="COSINE",
    params={ "nlist": 128 }
)

client.create_collection(
    collection_name="exp",
    schema=schema,
    index_params=index_params
)
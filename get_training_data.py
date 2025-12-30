# from datatrove.pipeline.readers import ParquetReader

# limit determines how many documents will be streamed (remove for all)
# to fetch a specific dump: hf://datasets/HuggingFaceFW/fineweb/data/CC-MAIN-2024-10
# replace "data" with "sample/100BT" to use the 100BT sample
# data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb/data", limit=1000)
# for document in data_reader():
#     # do something with document
#     print(document)

from datatrove.pipeline.filters import LambdaFilter
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            # replace "data/CC-MAIN-2024-10" with "sample/100BT" to use the 100BT sample
            ParquetReader("hf://datasets/HuggingFaceFW/fineweb/data/CC-MAIN-2024-10", limit=100000),
            LambdaFilter(lambda doc: doc.metadata["language"] == "en"),
            JsonlWriter("pretrain_data"),
        ],
        tasks=10,
    )
    pipeline_exec.run()

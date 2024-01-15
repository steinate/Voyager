import os

import voyager.utils as U
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.vectorstores import Chroma

from voyager.prompts import load_prompt
from voyager.control_primitives import load_control_primitives

def checkposition(pos1, pos2):
    if(abs(pos1['x'] - pos2['x']) + abs(pos1['y'] - pos2['y']) + abs(pos1['z'] - pos2['z']) < 3):
        return True
    else: 
        return False


class ResourceManager:
    def __init__(
        self,
        model_name="gpt-3.5-turbo",
        temperature=0,
        retrieval_top_k=5,
        request_timout=120,
        ckpt_dir="ckpt",
        resume=False,
    ):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            request_timeout=request_timout,
        )
        U.f_mkdir(f"{ckpt_dir}/resource/block")
        U.f_mkdir(f"{ckpt_dir}/resource/description")
        U.f_mkdir(f"{ckpt_dir}/resource/vectordb")
        # programs for env execution
        self.control_primitives = load_control_primitives()
        if resume:
            print(f"\033[33mLoading Resource Manager from {ckpt_dir}/resource\033[0m")
            self.resources = U.load_json(f"{ckpt_dir}/resource/resources.json")
        else:
            self.resources = {}
        self.retrieval_top_k = retrieval_top_k
        self.ckpt_dir = ckpt_dir
        self.vectordb = Chroma(
            collection_name="resource_vectordb",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=f"{ckpt_dir}/resource/vectordb",
        )
        assert self.vectordb._collection.count() == len(self.resources), (
            f"Resource Manager's vectordb is not synced with resources.json.\n"
            f"There are {self.vectordb._collection.count()} resources in vectordb but {len(self.resources)} resources in resources.json.\n"
            f"Did you set resume=False when initializing the manager?\n"
            f"You may need to manually delete the vectordb directory for running from scratch."
        )

    # @property
    # def programs(self):
    #     programs = ""
    #     for skill_name, entry in self.skills.items():
    #         programs += f"{entry['code']}\n\n"
    #     for primitives in self.control_primitives:
    #         programs += f"{primitives}\n\n"
    #     return programs

    def add_new_resource(self, pos, blocks):
        # both str
        resource_description = self.generate_resource_description(pos, blocks)

        if pos in self.resources:
            print(f"\033[33mpos {pos} already exists. Rewriting!\033[0m")
            self.vectordb._collection.delete(ids=[pos])
            i = 2
            while f"{pos}V{i}.js" in os.listdir(f"{self.ckpt_dir}/resource/block"):
                i += 1
            dumped_resource_name = f"{pos}V{i}"
        else:
            dumped_resource_name = pos
        self.vectordb.add_texts(
            texts=[resource_description],
            ids=[pos],
            metadatas=[{"name": pos}],
        )
        self.resources[pos] = {
            "block": blocks,
            "description": resource_description,
        }
        assert self.vectordb._collection.count() == len(
            self.resources
        ), "vectordb is not synced with resources.json"
        U.dump_text(
            blocks, f"{self.ckpt_dir}/resource/block/{dumped_resource_name}.js"
        )
        U.dump_text(
            resource_description,
            f"{self.ckpt_dir}/resource/description/{dumped_resource_name}.txt",
        )
        U.dump_json(self.resources, f"{self.ckpt_dir}/resource/resources.json")
        self.vectordb.persist()

    def generate_resource_description(self, pos, blocks):
        messages = [
            SystemMessage(content=load_prompt("resource")),
            HumanMessage(
                content=blocks
                + "\n\n"
                + f"The observating position is `{pos}`."
            ),
        ]
        block_description = self.llm(messages).content
        return f"{pos}(bot) {{\n{block_description}\n}}"

    def retrieve_resources(self, query):
        k = min(self.vectordb._collection.count(), self.retrieval_top_k)
        if k == 0:
            return []
        print(f"\033[33mResource Manager retrieving for {k} resources\033[0m")
        docs_and_scores = self.vectordb.similarity_search_with_score(query, k=k)
        print(
            f"\033[33mResource Manager retrieved resources: "
            f"{', '.join([doc.metadata['name'] for doc, _ in docs_and_scores])}\033[0m"
        )
        resources = []
        for doc, _ in docs_and_scores:
            resources.append((doc.metadata["name"], self.resources[doc.metadata["name"]]["block"]))
        return resources

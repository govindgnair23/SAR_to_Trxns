# filename: gpt4_paper_summary.py
with open("gpt4_paper_summary.txt", "w") as f:
    f.write("Latest Papers on GPT-4 from arXiv:\n\n")
    
    f.write("1. Title: Contrastive Learning to Improve Retrieval for Real-world Fact Checking\n")
    f.write("   Authors: Aniruddh Sriram, Fangyuan Xu, Eunsol Choi, Greg Durrett\n")
    f.write("   Summary: This paper presents a method for improved retriever performance in fact-checking that incorporates diverse evidence to assess the veracity of claims.\n\n")
    
    f.write("2. Title: ProtocoLLM: Automatic Evaluation Framework of LLMs on Domain-Specific Scientific Protocol Formulation Tasks\n")
    f.write("   Authors: Seungjun Yi, Jaeyoung Lim, Juyong Yoon\n")
    f.write("   Summary: This paper proposes an automatic framework for evaluating large language models' capabilities in generating scientific protocols.\n\n")
    
    f.write("Potential Applications in Software:\n")
    f.write("- Fact-Checking Applications: Automated tools for verifying information reliability in content.\n")
    f.write("- Scientific Research Tools: Software for automating and validating scientific protocols.\n")

print("Summary written in gpt4_paper_summary.txt")
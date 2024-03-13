# Medical_chatbot

We all search for medical ailments and terminologies out of curiosity or even for self-diagnosis (which is ofcourse a wrong thing to do!). The most reliable source for such queries is none other than a proper medical textbook (something which every Doctor has to read thoroughly). Its time taking and sometimes exhaustive to go through a giant textbook looking for a specific term. We can thus take help of Large Language Models (LLM) to learn the textbook information on our behalf and serve us an appropriate response. 

LLMs have been widely celebrated and applied to building custom models for different applications. LLMs are already pre-trained on a large corpus of text data inlcuding how to interact like a human. Thus, if such pre-trained models are taught a specific subject or topic, they quickly grasp its nuance and generate an almost human like response explaining those topics. The creativity of LLM models are controlled with a paremeter called 'temeprature' whose value ranges from 0 to 1. 

A higher temperature value allows the model to take risk and generate creative responses, however there is a chance to give wrong response as well. Whereas a lower temperature value keeps the model deterministic, the model doesn't take any risk and generates a safe response. 

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

Here I have trained one such open source LLM called Llama-2. It comes in different variants (trained on 3 billion, 7 billion, and 13 billion parameters). We can choose one based on our computational capacity and app requirements. I chose to go with 7 billion parameter (4 bit model). For those who have a GPU can go with a 8 bit model. A lower bit model allows the training to happen on CPU while higher bit requires GPU.

![llama-2](https://github.com/shazam37/Medical_chatbot/assets/119686545/861fadda-ed16-4bff-a2d6-64e731e69270)


I downloaded a medical textbook called Gale's encyclopaedia of medicine- Volume 1 (There are other volumes too but we need to buy them, however I also chose to keep it simple and went with limited data). The data was tokenized, embeddings were generated using HuggingFace library, semantic index was produced using the embeddings and finally stored on Pinecone vector DB as a knowledge graph (Pinecone is used for storing vector data). If the user sends in query then it is tokenized, and sent for QueryRetrieval on Pinecone DB. The Pinecone returns top 2 results (ranked results) similar to our query embeddings using cosine similarity (cosine similarity is a metric used for calculating similarity between two vectors). The ranked results are then passed through the Llama-2 which gives back an answer to our query. The flowchart for the process looks like:

![Model_Architecture](https://github.com/shazam37/Medical_chatbot/assets/119686545/98d661ff-974d-4413-a95b-c928999b8ae9)

The entire pipeline was finally wrapped up into a Flask application whose interface looks like:

![Screenshot from 2024-03-13 16-11-40](https://github.com/shazam37/Medical_chatbot/assets/119686545/845c4b62-aa32-466d-89a8-d856ad537461)

Lets say if we give a query: "What is Acne?"

![Screenshot from 2024-03-13 16-12-53](https://github.com/shazam37/Medical_chatbot/assets/119686545/abd38925-eb9c-49a6-89d0-966c200a293c)

We receive the output:

![Screenshot from 2024-03-13 16-14-31](https://github.com/shazam37/Medical_chatbot/assets/119686545/1847eb5b-9c90-48c0-97e6-ed20dd0783e4)

Next we ask about the medicine Acetaminophen and receive the response:

![Screenshot from 2024-03-13 16-33-28](https://github.com/shazam37/Medical_chatbot/assets/119686545/ed397c94-6727-4f01-bf88-7835123a8866)

Since the model is trained on less corpus of medical data so we many not get very detailed response. We can buy more medical textbooks and feed it into our database to expand the scope of this chat bot. But I feel its still a good starting point. We can build all sort of marvellous applications for our specific task given we have proper data. Its only the beginning that we have started leveraging the power of LLMs, and there is certainly a long way to go ahead! 

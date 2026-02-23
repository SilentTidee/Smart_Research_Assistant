from langserve import RemoteRunnable

summarize = RemoteRunnable("http://localhost:8000/summarize")

result = summarize.invoke({
    "text": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. AI research is defined as the study of intelligent agents: any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term 'artificial intelligence' is often used to describe machines that mimic cognitive functions that humans associate with the human mind, such as learning and problem-solving."
})
print(result)
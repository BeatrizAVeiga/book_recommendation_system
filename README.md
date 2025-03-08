# <img src="Streamlit\icons\FAVICON.png" alt="drawing" width="35"/>&nbsp; **Book Recommendation System** ✨


<img src="Multimedia\presentation.png" alt="drawing" width="1000"/>




## **Overview** 🌟

This project is a **Book Recommendation System** that leverages **Natural Language Processing (NLP)** and **BERTopic** for clustering similar books based on their descriptions. It allows users to input a description of a book they love, and the system will recommend books with similar topics using **SBERT embeddings** and **cosine similarity**. The goal of the system is to provide personalized book recommendations tailored to a user's input. 📖💡

The system is built as a **Streamlit application**, which allows users to interact with the recommendation model in a user-friendly and engaging interface. 🎉

### **Key Features** 🔑
- **Book Search & Recommendation**: Let users input a description of a book they enjoy and get recommended similar books. 
- **Topic Modeling**: Uses **BERTopic** for topic modeling to group books with similar themes or subjects. 
- **NLP Preprocessing**: Preprocessing is done to clean and standardize the input descriptions, ensuring better quality recommendations. 
- **Book Format Filter**: Users can filter the recommendations by book format, including **Paperback**, **Hardcover**, and **Ebook**. 
- **Rating Filter**: Recommendations are filtered based on a minimum rating threshold. ⭐

## **Technology Stack** 💻

- **Python** 
- **Streamlit**: Framework for building the interactive UI. 
- **BERTopic**: Used for topic modeling and clustering books based on their descriptions. 
- **SBERT**: Sentence-BERT embeddings are used to generate book description embeddings for similarity comparison. 
- **Pandas**
- **Scikit-learn**: For cosine similarity calculation. 
- **Requests**: For fetching book covers via ISBN from the Open Library API. 

## **How it Works** 🔍

### **1. Input Description** 📝
- The user inputs a description of a book they enjoy (e.g., "Books about LGBTQ+ Activism"). 
- The system uses **SBERT embeddings** to convert the input description into a vector representation, capturing the meaning and context of the words. 

### **2. SBERT Embedding & Cosine Similarity** 🤖
- The input text is embedded using **SBERT** to convert it into a vector that represents the semantic meaning of the description.
- **Cosine Similarity** is then used to measure how similar the user's input is to the descriptions of books in the dataset. 

### **3. Topic Modeling with BERTopic** 📊
- The system uses **BERTopic** to assign topics to the books in the dataset.
- Based on the user's input, the system finds the most relevant topic and filters books belonging to the same topic. 
- Books are clustered into topics, allowing for more meaningful and relevant recommendations.

### **4. Book Recommendations** 📈
- The filtered recommendations are sorted by their **cosine similarity** to the user's input. 📖
- Users can further filter the recommendations based on their preferred **book format** (e.g., paperback, ebook) and **minimum rating**. ⭐
  
### **5. Display Book Details** 🖼️
- The system fetches and displays book details such as the title, author, description, rating, format, and cover image. 🏷️📸
  
## **Demo Video** 🎥

- You can find the Demo Video in the Multimedia folder!

## **Presentation Slides** 📸

- Feel free to access the presentation online through the [following link](https://www.canva.com/design/DAGggpgv-MA/Jm_z_7i8kwG8evZV4jZUjw/view?utm_content=DAGggpgv-MA&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h50b13c1749), or download it directly from the repository. <br>
**Please note that the presentation includes a video, which can only be viewed through the online link.** 😊

## **Datasets** 📂
- You can access the datasets I used through the link below: <br>
🔗 [Books Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html#datasets:~:text=goodreads_books_history_biography.json.gz) <br>
🔗 [Authors Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html#datasets:~:text=Detailed%20information%20of%20authors%3A%20goodreads_book_authors.json.gz)

## **License** 📄
- This project is licensed under the MIT License - see the [LICENSE file](https://github.com/BeatrizAVeiga/book_recommendation_system/blob/main/LICENSE) for details. ⚖️
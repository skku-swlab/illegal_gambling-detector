### **[Tracking Illegal Gambling Website]** (2024.10 ~ 2025.7)  **[Selected 2025-1 SKKU CO-Deeplearning]** 
The operation of illegal gambling websites poses a serious threat to our society.  
However, manually identifying and blocking these sites is highly impractical.  
To address this challenge, our SWLab has developed an AI-powered tracking system for detecting illegal gambling websites.  

🕸️ Web Crawling Modules  
1. OCR-Based Banner Image Analysis  
Illegal gambling websites often use promotional banners for advertising. We apply OCR (Optical Character Recognition) technology to detect such banners and extract associated URLs for further crawling.  

2. Redirection Analysis  
Operators of illegal gambling sites often hack legitimate websites and set up redirections that reroute visitors to illegal destinations. Our system identifies and analyzes such hidden redirections.  

3. SMS Spam Dataset Analysis  
We utilize the spam SMS dataset provided by KISA to periodically extract and update known illegal gambling URLs.  

4. Social Media Crawling  
We crawl promotional posts from platforms such as Instagram, X (formerly Twitter), and Facebook to identify and collect illegal gambling URLs.  

5. Telegram sLLM Chatbot [LLM Spear Phishing Chatbot] 
To ensure more precise enforcement, blocking must go beyond URLs to include associated bank account numbers. We are developing a specialized sLLM-based Telegram chatbot capable of extracting bank account information from chat messages. 
<div align='center'>
<img width="790" alt="image" src="https://github.com/user-attachments/assets/8db21a56-6e9e-468d-9397-1cf5662ff86e" />
</div>

🤖 AI-Based Classification Modules  
1. Image Classification  
Illegal gambling websites tend to share visually similar design elements. We classify these images using CNN-based architectures such as EfficientNet and ResNet.  

2. HTML Structure Analysis  
We apply our newly developed BAG (BERT-Attention GNN) architecture to effectively analyze and classify the HTML structure of websites for illegal gambling indicators.  

<div align="center">
  <img width="480" alt="Screenshot 2025-04-29 at 10 48 46 AM" src="https://github.com/user-attachments/assets/e258c7f3-6868-4539-a3f1-440ee5aca357" />
</div>  

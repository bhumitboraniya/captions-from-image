# Automated Image Captioning using Neural Networks  
**Submitted By**  
Priyank Vaghela (22BCE544)  
Bhumit Boraniya (22BCE503)  

**Department of Computer Science and Engineering**  
Institute of Technology, Nirma University, Ahmedabad  
November 2024  

---

## Certificate  
This is to certify that the minor project entitled *“Automated Image Captioning using Neural Networks”* submitted by Priyank Vaghela (22BCE544) & Bhumit Boraniya (22BCE503) is the original work carried out under the supervision of **Dr. Ramesh Ram Naik**. The results embodied in this project have not been submitted to any other institution for academic credit.  

**Dr. Ramesh Ram Naik**  
*Professor, CSE Department*  

**Dr. Sudeep Tanwar**  
*Professor and Head, CSE Department*  

---

## Statement of Originality  
We hereby declare that this project is our original work. No part of this project has been plagiarized or submitted elsewhere for any degree/diploma.  

**Signatures**  
Priyank Vaghela | Date:  
Bhumit Boraniya | Date:  
*Endorsed by Dr. Ramesh Ram Naik*  

---

## Acknowledgements  
We thank **Dr. Ramesh Ram Naik** for his guidance, Nirma University for resources, and our peers/family for their support during this project.  

---

## Abstract  
This project combines **CNNs** (for image feature extraction) and **Transformers** (for sequence modeling) to generate contextual captions for images. Trained on the **COCO dataset**, the model achieves high scores on BLEU, METEOR, and CIDEr metrics. Applications include assistive technology and automated content generation.  

---

## Abbreviations  
| Abbreviation | Full Form |  
|--------------|-----------|  
| CNN | Convolutional Neural Network |  
| COCO | Common Objects in Context |  
| BLEU | Bilingual Evaluation Understudy |  
| LSTM | Long Short-Term Memory |  
| GAN | Generative Adversarial Network |  
| CIDEr | Consensus-based Image Description Evaluation |  

---

## Methodology  
### Key Steps:  
1. **Data Preprocessing**:  
   - Clean and integrate the COCO dataset.  
   - Tokenize captions and preprocess images (resizing, normalization).  

2. **Model Architecture**:  
   - **Encoder**: Pre-trained CNN (e.g., InceptionV3) for feature extraction.  
   - **Decoder**: Transformer model to generate captions from embeddings.  

3. **Training**:  
   - Loss Function: Sparse Categorical Cross-Entropy.  
   - Optimizer: Adam.  
   - Metrics: BLEU, METEOR, CIDEr.  

4. **Inference**:  
   - Generate captions for unseen images using beam search.  

---

## Results  
- **Training Loss**: Decreased consistently over 10 epochs (see *Figure 3.4*).  
- **Caption Quality**: Achieved BLEU-4 score of **0.35** and CIDEr score of **0.85**.  
- **Real-Time Performance**: Caption generation in <500ms per image.  

### Figures:  
1. **Encoder-Decoder Architecture**  
   ![Encoder](images/encoder.png) *Figure 3.1*  
   ![Decoder](images/decoder.png) *Figure 3.2*  

2. **Caption Length Distribution**  
   ![Caption Lengths](images/caption_length_distribution.png) *Figure 3.3*  

3. **Training vs Validation Loss**  
   ![Loss Graph](images/model_loss.png) *Figure 3.4*  

---

## Code Implementation  
### Algorithm: Image Captioning Pipeline  
```python  
# Step 1: Load and preprocess COCO dataset  
dataset = load_coco_data(path="data/coco")  

# Step 2: Build Model  
encoder = CNN_Encoder()  
decoder = Transformer_Decoder()  

# Step 3: Train  
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")  
model.fit(dataset, epochs=10)  

# Step 4: Generate Captions  
caption = generate_caption(image_path="test_image.jpg")  
print(caption)  

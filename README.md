# Predicting And Segmenting Churned Users In Marketing
Using Machine Learning to predict user churn then segment these users into groups.

## **I. Introduction**

### **1. Situation**

- One ecommerce company has a project on **predicting churned users** and **segment users** in order to **offer potential promotions**.
  
- Data Dictionary:

  ![Data Dictionary](https://github.com/user-attachments/assets/fcbc7520-45a2-4263-92cd-cb1ca220df47)

### **2. Task**

- Build the Machine Learning model for predicting churned users. Choose six Feature Importance from the model that they are the behaviors of churned users. 
 
- Based on six Feature Importance, segment these churned users into groups so that the company offers some special promotions for them.

## **II. Supervised Learning**

### **1. EDA**


### **2. Transforms features**


### **3. Apply Random Forest model**


### **4. Choose top 6 features importance to analyze**


## **III. Unsupervised Learning**

### **1. Dimension Reduction**


### **2. Apply K-Means model**


### **3. GridSearchCV model**


## **IV. Conclusion**
The majority of churned users are:
 - Using Mobile Phone
 - Tenure in ranges from 0 to 20
 - DaySinceLastOrder in ranges from 0 to 20 recent days
 - CashbackAmount	in ranges from 100 to 325

 => According to the distribution, the order from most to few users in the cluster is 1-3-2-0.
1. Cluster 1:
  - CashbackAmount: Large (about 190-250)
2. Cluster 3:
  - CashbackAmount: Smallest (about 100-140)
3. Cluster 2:
  - CashbackAmount: Small (about 140-190)
4. Cluster 0:
  - CashbackAmount: Largest (about >250)

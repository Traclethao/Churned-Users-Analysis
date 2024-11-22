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
#### **1.1. Handle Missing/Duplicate/Incorrect Values** 
- Check

  ![Check](https://github.com/user-attachments/assets/00d813d5-ddc7-4c89-98c7-6efa26e6d46b)

  => Six isna features are numeric columns (Tenure, HourSpendOnApp, CouponUsed, DaySinceLastOrder, OrderAmountHikeFromlastYear, OrderCount, WarehouseToHome)
  + Tenure, HourSpendOnApp, CouponUsed, DaySinceLastOrder have 0 in values => replace 0.
  + OrderAmountHikeFromlastYear do not have 0 in values but it is a percentage increase in order from last year. If the number orders last year = the number orders in this year => can have 0 in values => replace 0.
  + OrderCount do not have 0 in values but it is Total number of orders has been places in last month. If the customer did not buy anything last month => can have 0 in values => replace 0.
  + WarehouseToHome  => cann't have 0 in values => check outlier => WarehouseToHome cover outliers => replace median
 
    ![WarehouseToHome](https://github.com/user-attachments/assets/c114eb45-5909-4822-bc32-be2464f04ea0)
  
- Handle numeric columns
  + Handle when features have 0 in values => replace 0
 
    ![Replace 0](https://github.com/user-attachments/assets/9a9638d7-eeb2-4cfa-afad-bc04bf07b0c1)
  
  + Handle when features cover have outliers => replace Median
 
    ![Replace Median](https://github.com/user-attachments/assets/3225fc91-9e99-4700-aec5-27acce9dbc8a)

#### **1.2. Check imbalanced**

![Check imbalanced](https://github.com/user-attachments/assets/044ea56e-d759-4648-a03d-af89fb174c48)

=> The ratio of label 1 on total is 16% => We can continue with the EDA and ML model 

#### **1.3. Univariate Analysis** 

![Univariate Analysis](https://github.com/user-attachments/assets/2c4d34e8-d182-472a-8a6c-713f4cdbd06a)
![Univariate Analysis 1](https://github.com/user-attachments/assets/e7353d2a-80c4-4e46-9f1a-fe8f0cc8e544)

As shown the unique values of each numeric column, there are 5 columns that have low unique values (less than 10 values), which are CityTier, HourSpendOnApp, NumberOfDeviceRegistered, SatisfactionScore and Complain.
  => HourSpendOnApp, NumberOfDeviceRegistered, SatisfactionScore, Complain have numeric dtype but don't have category meaning.
  => CityTier has numeric dtype but has category meaning. => convert to object

  ![Univariate Analysis result](https://github.com/user-attachments/assets/d8d44587-bb3e-438a-bf11-47de56fe3e9f)

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

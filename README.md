# Predicting And Segmenting Churned Users In Marketing
Using Machine Learning to predict user churn then segment these users into groups.

## **I. Introduction**

### **1. Situation**

- One ecommerce company has a project on **predicting churned users** and **segment users** in order to **offer potential promotions**.

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

![Transforms Features](https://github.com/user-attachments/assets/04ee1670-5e85-4655-b0b6-734fee86905f)

### **3. Apply Random Forest model**
- Split train/test set

  ![Split traintest set - SL](https://github.com/user-attachments/assets/419bf1b0-0782-4f1c-b271-763787a45a8f)
  
- Normalization

  ![Normalization - SL](https://github.com/user-attachments/assets/5f57e0c9-a256-497d-9607-e4ccd28de680)

- Random Forest Classifier

  ![Random Forest Classifier](https://github.com/user-attachments/assets/4002850e-9c1d-4895-b7ec-9973263ca0cd)

=> The accuracy index on the test set is 94% that means this model has correct predictions and correct features importance. 

### **4. Choose top 6 features importance to analyze**

- Show Feature Importance from model.

  ![shown features importance](https://github.com/user-attachments/assets/07ce63af-c883-4950-a982-a3a24d1479ca)

  ![result FI](https://github.com/user-attachments/assets/5c8a9bdc-493d-44e4-9584-02eb0ae49cd0)

- From the Feature Important chart, choose the 6 highest features: Tenure, Complain, DaySinceLastOrder, CashbackAmount, MaritalStatus_Single, PreferedOrderCat_Mobile Phone. Get churned users. 

  ![choose 6 FI](https://github.com/user-attachments/assets/5aff37c2-c9cb-4c45-8672-146670ecafe9)

  ![6FI](https://github.com/user-attachments/assets/2a2d1b45-c96e-40c3-9175-ab81d22b1322)

## **III. Unsupervised Learning**

### **1. Dimension Reduction**

![Dimension Reduction 1](https://github.com/user-attachments/assets/f15f85fe-58f8-4784-bfbd-662f42fc6cb1)

![Dimension Reduction 2](https://github.com/user-attachments/assets/84a368e6-a946-4002-b96c-67bb98084a6a)

### **2. Apply K-Means model**

#### **2.1. Choose K**

  ![Choose K](https://github.com/user-attachments/assets/246702b1-194d-4c11-867d-8cc0e282916c)
  
  => Choose k=4
  
#### **2.2. Apply K-Means**

  ![Apply K-Means](https://github.com/user-attachments/assets/6824fde2-80af-459e-8cd0-085ed3319f9d)

#### **2.3. Silhouette Score** 

  ![Silhouette Score](https://github.com/user-attachments/assets/1cc73fcc-38ed-424c-81b9-c63048fdfffd)

  => silhouette_score = 0.5, shows that the data is clustered quite well. 
  
#### **2.4. Distribution of clusters**

- Distribution Of The Clusters

  ![Clusters](https://github.com/user-attachments/assets/fa4b98f8-b8f7-4569-81b4-d8b7021bb9ad)
  
  => Cluster 1 has the most distribution (accounted for more 400 users)
  
- Cluster's Profile Based On CashbackAmount And Tenure

  ![CashbackAmount And Tenure](https://github.com/user-attachments/assets/e57339da-6190-4376-b6ba-53236ff9d883)

  => The distribution of clusters by CashbackAmount And Tenure is almost the same.

- Cluster's Profile Based On CashbackAmount And DaySinceLastOrder

  ![CashbackAmount And DaySinceLastOrder](https://github.com/user-attachments/assets/489e53fc-bfb9-4a85-8b5e-21e518ab98a0)

  => The distribution of clusters is mainly concentrated in the range of 0-20 recent days.

- Cluster's Profile Based On MaritalStatus

  ![MaritalStatus](https://github.com/user-attachments/assets/067979fa-0585-4c23-8114-32d6d7319796)
  
- Cluster's Profile Based On Complain

  ![Complain](https://github.com/user-attachments/assets/952bd478-0e09-4bb2-b8ed-d833d3bc94ea)
  
- Cluster's Profile Based On PreferedOrderCat

  ![PreferedOrderCat](https://github.com/user-attachments/assets/6f02d2bb-d727-49d3-a4e4-2760438b2460)
  
### **3. GridSearchCV model**

- Split train/test set & Normalization

  ![Split traintest set   Normalization](https://github.com/user-attachments/assets/f0abf158-4e1a-4a09-941a-29bc3537c35f)
  
- Apply Model

  ![Apply Model](https://github.com/user-attachments/assets/41a9a16a-3ec9-40ab-874c-5f23c843bac9)
  
- Feature Importance of Clusters

  ![Feature Importance of Clusters](https://github.com/user-attachments/assets/d0281212-b4a9-48d0-842b-389d5d0acffb)

  ![Feature Importance of Clusters - result](https://github.com/user-attachments/assets/4a996e92-7a76-4572-852e-c1533ddaa0b3)

  => CashbackAmount is the most important factor in clustering for churned users. It is 3 times more important than Tenure, which is the second most important.

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

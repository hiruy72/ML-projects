"""
Simple Classification with RandomForestClassifier
Following the 6-step ML workflow
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("🌸 Iris Classification with Random Forest")
    print("=" * 45)
    
    # Step 1: Load and explore data
    print("Step 1: Loading Iris dataset...")
    iris = load_iris()
    X = iris.data  # Features: sepal length, sepal width, petal length, petal width
    y = iris.target  # Target: setosa, versicolor, virginica
    
    print(f"Dataset: {X.shape[0]} flowers, {X.shape[1]} features")
    print(f"Classes: {list(iris.target_names)}")
    print(f"Features: {list(iris.feature_names)}")
    
    # Step 2: Data preprocessing (Iris is already clean!)
    print("\nStep 2: Data preprocessing...")
    print("✅ Iris dataset is clean - no missing values or categorical variables")
    
    # Step 3: Split the data
    print("\nStep 3: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training: {len(X_train)} samples")
    print(f"Testing: {len(X_test)} samples")
    
    # Step 4: Train the model
    print("\nStep 4: Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model trained successfully!")
    
    # Step 5: Make predictions
    print("\nStep 5: Making predictions...")
    y_pred = model.predict(X_test)
    print(f"Predicted {len(y_pred)} flower species")
    
    # Step 6: Evaluate performance
    print("\nStep 6: Evaluating model...")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy: {accuracy:.2%}")
    
    print("\n📊 Detailed Results:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Bonus: Feature importance
    print("🔍 Most Important Features:")
    importances = model.feature_importances_
    for i, (feature, importance) in enumerate(zip(iris.feature_names, importances)):
        print(f"  {i+1}. {feature}: {importance:.3f}")
    
    print("\n🎉 Classification completed successfully!")
    print(f"The model achieved {accuracy:.1%} accuracy on the test set.")

if __name__ == "__main__":
    main()
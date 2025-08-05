loss_values, features_per_epoch = train_dcjdan(epochs, learning_rate, batch_size)

for epoch, (loss, (f_s, f_t)) in enumerate(zip(loss_values, features_per_epoch)):
    if epoch % 5 == 0:
        from sklearn.manifold import TSNE
        combined_features = torch.cat([f_s, f_t])
        reduced = TSNE(n_components=2).fit_transform(combined_features)
        plt.figure()
        plt.scatter(reduced[:, 0], reduced[:, 1])
        plt.title(f"t-SNE at epoch {epoch+1}")
        st.pyplot(plt)

plt.figure()
plt.plot(range(1, len(loss_values)+1), loss_values, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
st.pyplot(plt)

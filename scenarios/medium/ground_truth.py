// ground_truth.js
async function getUser(userId) {
    try {
        const response = await fetch(`https://api.example.com/users/${userId}`);
        if (!response.ok) {
            throw new Error(`Failed: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        throw error;
    }
}

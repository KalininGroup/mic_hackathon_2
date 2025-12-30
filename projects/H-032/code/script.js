const form = document.getElementById("chat-form");
const chatBox = document.getElementById("chat-box");
const resultsBox = document.getElementById("results");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const prompt = document.getElementById("prompt").value;
    const image = document.getElementById("image").files[0];

    // Add user message
    chatBox.innerHTML += `
        <div class="user">
            <span>🧑‍🔬</span>
            <div>${prompt}</div>
        </div>
    `;
    chatBox.scrollTop = chatBox.scrollHeight;

    const formData = new FormData();
    formData.append("prompt", prompt);
    formData.append("image", image);

    // Show loading indicator
    chatBox.innerHTML += `
        <div class="bot loading">
            🤖 Analyzing image and segmentation...
        </div>
    `;
    chatBox.scrollTop = chatBox.scrollHeight;

    const res = await fetch("/chat", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    // Remove loading
    document.querySelector(".loading").remove();

    // Populate results
    document.getElementById("input-image").src = data.input_image_url
    document.getElementById("count").innerText = data.stats.count;
    document.getElementById("area").innerText = data.stats.mean_area;
    document.getElementById("overlay").src = data.overlay_url
    document.getElementById("response").innerText = data.explanation;

    resultsBox.classList.remove("hidden");

    // Add bot message
    chatBox.innerHTML += `
        <div class="bot">
            <span>🤖</span>
            <div>${data.explanation}</div>
        </div>
    `;
    chatBox.scrollTop = chatBox.scrollHeight;
});

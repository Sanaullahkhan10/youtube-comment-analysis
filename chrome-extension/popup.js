// popup.js
// --------------------------------------------------
// This script runs when the extension popup opens.
// It does 5 things:
// 1. Finds the YouTube video ID from the current tab URL
// 2. Fetches comments for that video from the YouTube Data API
// 3. Sends those comments to our sentiment analysis API
// 4. Displays each comment along with its predicted sentiment
// 5. Fetches and displays a pie chart, word cloud, and trend graph
//
// Note: YOUTUBE_API_KEY and SENTIMENT_API_BASE_URL come from
// config.js, which is loaded before this file (see popup.html).
// --------------------------------------------------

// Get references to the boxes in popup.html
var statusBox = document.getElementById("status");
var resultsBox = document.getElementById("results");
var visualsBox = document.getElementById("visuals");

// This function fetches comments for a given YouTube video ID.
// It returns a list of {text, timestamp} objects.
async function fetchYoutubeComments(videoId) {
  var url = "https://www.googleapis.com/youtube/v3/commentThreads"
    + "?part=snippet"
    + "&videoId=" + videoId
    + "&maxResults=50"
    + "&order=relevance"
    + "&key=" + YOUTUBE_API_KEY;

  var response = await fetch(url);
  var data = await response.json();

  if (data.error) {
    throw new Error(data.error.message);
  }

  var comments = [];

  for (var i = 0; i < data.items.length; i++) {
    var item = data.items[i];
    var snippet = item.snippet.topLevelComment.snippet;

    var commentText = snippet.textOriginal;
    var commentTimestamp = snippet.publishedAt;

    comments.push({ text: commentText, timestamp: commentTimestamp });
  }

  return comments;
}

// This function sends the comments to our FastAPI backend and
// gets back each comment's predicted sentiment.
async function analyzeSentiment(comments) {
  var url = SENTIMENT_API_BASE_URL + "/predict_with_timestamps";

  var response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ comments: comments })
  });

  var results = await response.json();
  return results;
}

// This is a generic helper that POSTs a JSON payload to one of our
// image-generating endpoints, and turns the returned PNG bytes into
// a URL the browser can use as an <img> source.
async function fetchImageFromApi(endpointPath, payload) {
  var url = SENTIMENT_API_BASE_URL + endpointPath;

  var response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  var imageBlob = await response.blob();
  var imageUrl = URL.createObjectURL(imageBlob);
  return imageUrl;
}

// This function displays the text results inside the popup.
function displayResults(results) {
  resultsBox.innerHTML = "";

  // Count how many comments fall into each sentiment category
  var positiveCount = 0;
  var neutralCount = 0;
  var negativeCount = 0;

  for (var i = 0; i < results.length; i++) {
    var sentiment = results[i].sentiment;

    if (sentiment === "1") {
      positiveCount = positiveCount + 1;
    } else if (sentiment === "0") {
      neutralCount = neutralCount + 1;
    } else if (sentiment === "-1") {
      negativeCount = negativeCount + 1;
    }
  }

  var summaryText = "Positive: " + positiveCount + " | Neutral: " + neutralCount + " | Negative: " + negativeCount;

  var summaryElement = document.createElement("p");
  summaryElement.textContent = summaryText;
  summaryElement.style.fontWeight = "bold";
  resultsBox.appendChild(summaryElement);

  // Show each individual comment with its sentiment label
  for (var i = 0; i < results.length; i++) {
    var comment = results[i].comment;
    var sentiment = results[i].sentiment;

    var sentimentLabel = "Neutral";
    var sentimentColor = "gray";

    if (sentiment === "1") {
      sentimentLabel = "Positive";
      sentimentColor = "green";
    } else if (sentiment === "-1") {
      sentimentLabel = "Negative";
      sentimentColor = "red";
    }

    var commentElement = document.createElement("div");
    commentElement.style.borderBottom = "1px solid #ddd";
    commentElement.style.padding = "4px 0";

    var commentTextElement = document.createElement("span");
    commentTextElement.textContent = comment;
    commentTextElement.style.fontSize = "12px";

    var sentimentTag = document.createElement("span");
    sentimentTag.textContent = " [" + sentimentLabel + "]";
    sentimentTag.style.color = sentimentColor;
    sentimentTag.style.fontWeight = "bold";
    sentimentTag.style.fontSize = "11px";

    commentElement.appendChild(commentTextElement);
    commentElement.appendChild(sentimentTag);
    resultsBox.appendChild(commentElement);
  }

  // Return the counts so the caller can reuse them for the pie chart
  return {
    positiveCount: positiveCount,
    neutralCount: neutralCount,
    negativeCount: negativeCount
  };
}

// This function fetches all three visualizations and adds them to the popup.
async function displayVisuals(rawComments, results, sentimentCounts) {
  visualsBox.innerHTML = "";

  // --- Pie chart ---
  var chartHeading = document.createElement("h3");
  chartHeading.textContent = "Sentiment Breakdown";
  visualsBox.appendChild(chartHeading);

  var chartPayload = {
    sentiment_counts: {
      "1": sentimentCounts.positiveCount,
      "0": sentimentCounts.neutralCount,
      "-1": sentimentCounts.negativeCount
    }
  };
  var chartUrl = await fetchImageFromApi("/generate_chart", chartPayload);
  var chartImage = document.createElement("img");
  chartImage.src = chartUrl;
  visualsBox.appendChild(chartImage);

  // --- Word cloud ---
  var wordcloudHeading = document.createElement("h3");
  wordcloudHeading.textContent = "Word Cloud";
  visualsBox.appendChild(wordcloudHeading);

  // Build a plain list of comment text strings (word cloud does not need timestamps)
  var commentTexts = [];
  for (var i = 0; i < rawComments.length; i++) {
    commentTexts.push(rawComments[i].text);
  }

  var wordcloudUrl = await fetchImageFromApi("/generate_wordcloud", { comments: commentTexts });
  var wordcloudImage = document.createElement("img");
  wordcloudImage.src = wordcloudUrl;
  visualsBox.appendChild(wordcloudImage);

  // --- Trend graph ---
  var trendHeading = document.createElement("h3");
  trendHeading.textContent = "Sentiment Trend Over Time";
  visualsBox.appendChild(trendHeading);

  // Build the {timestamp, sentiment} list, converting sentiment strings to numbers
  var sentimentData = [];
  for (var i = 0; i < results.length; i++) {
    sentimentData.push({
      timestamp: results[i].timestamp,
      sentiment: parseInt(results[i].sentiment, 10)
    });
  }

  var trendUrl = await fetchImageFromApi("/generate_trend_graph", { sentiment_data: sentimentData });
  var trendImage = document.createElement("img");
  trendImage.src = trendUrl;
  visualsBox.appendChild(trendImage);
}

// Main logic - runs as soon as the popup opens
async function main() {
  // Step 1: Find the current tab's URL
  var tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  var currentTab = tabs[0];
  var currentUrl = currentTab.url;

  // Step 2: Check if it is a YouTube video page
  var isYoutubeVideo = currentUrl.indexOf("youtube.com/watch") !== -1;

  if (!isYoutubeVideo) {
    statusBox.textContent = "Please open a YouTube video to use this extension.";
    return;
  }

  // Step 3: Extract the video ID from the URL
  var urlObject = new URL(currentUrl);
  var videoId = urlObject.searchParams.get("v");

  if (!videoId) {
    statusBox.textContent = "Could not find a video ID in this URL.";
    return;
  }

  statusBox.textContent = "Fetching comments...";

  try {
    // Step 4: Fetch comments from YouTube
    var comments = await fetchYoutubeComments(videoId);

    if (comments.length === 0) {
      statusBox.textContent = "No comments found for this video.";
      return;
    }

    statusBox.textContent = "Analyzing " + comments.length + " comments...";

    // Step 5: Send comments to our sentiment API
    var results = await analyzeSentiment(comments);

    // Step 6: Show the text results, and get back the sentiment counts
    statusBox.textContent = "Analysis complete for video: " + videoId;
    var sentimentCounts = displayResults(results);

    // Step 7: Fetch and show the pie chart, word cloud, and trend graph
    statusBox.textContent = "Generating visuals...";
    await displayVisuals(comments, results, sentimentCounts);
    statusBox.textContent = "Analysis complete for video: " + videoId;

  } catch (error) {
    statusBox.textContent = "Error: " + error.message;
  }
}

// Run the main function as soon as the popup loads
main();

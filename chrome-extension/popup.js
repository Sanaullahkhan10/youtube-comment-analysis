// popup.js
// --------------------------------------------------
// This script runs when the extension popup opens.
// Right now it only does one thing: check if the
// current browser tab is a YouTube video page, and if
// so, extract the video ID from the URL.
// --------------------------------------------------

// Get a reference to the status text box in popup.html
var statusBox = document.getElementById("status");

// Ask Chrome for information about the currently active tab
chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
  var currentTab = tabs[0];
  var currentUrl = currentTab.url;

  // Check if the URL looks like a YouTube watch page
  var isYoutubeVideo = currentUrl.indexOf("youtube.com/watch") !== -1;

  if (!isYoutubeVideo) {
    statusBox.textContent = "Please open a YouTube video to use this extension.";
    return;
  }

  // Extract the video ID from the URL.
  // A YouTube video URL looks like: https://www.youtube.com/watch?v=ABC123XYZ
  // The video ID is the value of the "v" query parameter.
  var urlObject = new URL(currentUrl);
  var videoId = urlObject.searchParams.get("v");

  if (!videoId) {
    statusBox.textContent = "Could not find a video ID in this URL.";
    return;
  }

  statusBox.textContent = "Video detected! ID: " + videoId;
});

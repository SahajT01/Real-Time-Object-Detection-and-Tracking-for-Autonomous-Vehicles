import React from 'react';

const VideoPlayer = ({ src, title }) => {
  return (
    <div>
      <h3>{title}</h3>
      <video width="600" controls>
        <source src={src} type="video/mp4" />
        Your browser does not support the video tag.
      </video>
    </div>
  );
};

export default VideoPlayer;

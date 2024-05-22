const VideoPlayer = ({ src, title }) => (
  <div style={{ marginTop: '20px' }}>
    <h3>{title}</h3>
    <video controls preload="auto" width="600">
      <source src={src} type="video/mp4" />
      Your browser does not support the video tag.
    </video>
  </div>
);

export default VideoPlayer;

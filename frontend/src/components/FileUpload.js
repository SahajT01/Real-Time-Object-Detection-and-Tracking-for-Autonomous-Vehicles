import React, { useState } from 'react';
import axios from 'axios';
import VideoPlayer from './VideoPlayer';

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [inputVideo, setInputVideo] = useState('');
  const [outputVideo, setOutputVideo] = useState('');

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setInputVideo(response.data.input_video);
      setOutputVideo(response.data.output_video);
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  return (
    <div>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload and Process</button>
      {inputVideo && <VideoPlayer src={inputVideo} title="Input Video" />}
      {outputVideo && <VideoPlayer src={outputVideo} title="Output Video" />}
    </div>
  );
};

export default FileUpload;

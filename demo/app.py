"""Interactive Streamlit demo for temporal graph neural networks."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data import TemporalGraphDataset, TemporalDataLoader, create_temporal_graph_data
from src.models import EvolveGCN, TGAT, TGN, DyRep, TemporalGNNEnsemble
from src.utils import get_device, set_seed


# Page configuration
st.set_page_config(
    page_title="Temporal Graph Neural Networks Demo",
    page_icon="🕒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .model-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset(dataset_name: str, root: str = "data"):
    """Load dataset with caching."""
    try:
        dataset = TemporalGraphDataset(root=root, name=dataset_name)
        return dataset
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


@st.cache_data
def generate_synthetic_data(num_nodes: int, num_edges: int, num_timesteps: int):
    """Generate synthetic temporal graph data."""
    return create_temporal_graph_data(
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_features=10,
        num_classes=2,
        num_timesteps=num_timesteps,
        seed=42
    )


def create_model(model_name: str, config: dict):
    """Create model based on configuration."""
    if model_name == "EvolveGCN":
        return EvolveGCN(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            hidden_channels=config["hidden_channels"],
            num_layers=config["num_layers"],
            dropout=config["dropout"]
        )
    elif model_name == "TGAT":
        return TGAT(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            hidden_channels=config["hidden_channels"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            time_dim=config["time_dim"]
        )
    elif model_name == "TGN":
        return TGN(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            hidden_channels=config["hidden_channels"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            time_dim=config["time_dim"]
        )
    elif model_name == "DyRep":
        return DyRep(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            hidden_channels=config["hidden_channels"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            time_dim=config["time_dim"]
        )
    elif model_name == "Ensemble":
        return TemporalGNNEnsemble(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            hidden_channels=config["hidden_channels"],
            dropout=config["dropout"]
        )


def visualize_graph_structure(snapshot, max_nodes: int = 100):
    """Visualize graph structure."""
    if snapshot.num_nodes > max_nodes:
        # Sample nodes for visualization
        node_indices = torch.randperm(snapshot.num_nodes)[:max_nodes]
        edge_mask = torch.isin(snapshot.edge_index[0], node_indices) & torch.isin(snapshot.edge_index[1], node_indices)
        edge_index = snapshot.edge_index[:, edge_mask]
        
        # Remap node indices
        node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)}
        edge_index = torch.tensor([[node_map[edge_index[0, i].item()], node_map[edge_index[1, i].item()]] 
                                 for i in range(edge_index.size(1))]).t()
        
        x = snapshot.x[node_indices]
        y = snapshot.y[node_indices]
    else:
        edge_index = snapshot.edge_index
        x = snapshot.x
        y = snapshot.y
    
    # Create node positions using spring layout
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from(edge_index.t().numpy())
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Extract coordinates
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=[f"Node {i}<br>Class: {y[i].item()}" for i in range(len(node_x))],
        marker=dict(
            size=10,
            color=y.numpy(),
            colorscale='Viridis',
            line=dict(width=2, color='black')
        )
    )
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Temporal Graph Structure',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Node colors represent classes",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor='left', yanchor='bottom',
                           font=dict(color="black", size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    return fig


def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">🕒 Temporal Graph Neural Networks Demo</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Dataset selection
    st.sidebar.subheader("Dataset")
    dataset_name = st.sidebar.selectbox(
        "Select Dataset",
        ["elliptic", "synthetic"],
        help="Choose between real Elliptic Bitcoin dataset or synthetic data"
    )
    
    # Model selection
    st.sidebar.subheader("Model")
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["EvolveGCN", "TGAT", "TGN", "DyRep", "Ensemble"],
        help="Choose the temporal GNN model to explore"
    )
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    hidden_channels = st.sidebar.slider("Hidden Channels", 16, 128, 64)
    num_layers = st.sidebar.slider("Number of Layers", 1, 5, 2)
    dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.1)
    
    # Model-specific parameters
    if model_name == "TGAT":
        num_heads = st.sidebar.slider("Number of Attention Heads", 1, 8, 4)
        time_dim = st.sidebar.slider("Time Dimension", 8, 32, 16)
    elif model_name in ["TGN", "DyRep"]:
        time_dim = st.sidebar.slider("Time Dimension", 8, 32, 16)
    else:
        num_heads = 4
        time_dim = 16
    
    # Data parameters
    st.sidebar.subheader("Data Parameters")
    if dataset_name == "synthetic":
        num_nodes = st.sidebar.slider("Number of Nodes", 100, 2000, 1000)
        num_edges = st.sidebar.slider("Number of Edges", 500, 10000, 5000)
        num_timesteps = st.sidebar.slider("Number of Timesteps", 10, 100, 50)
    else:
        num_nodes = 1000
        num_edges = 5000
        num_timesteps = 50
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "🔍 Graph Visualization", "🤖 Model Analysis", "📈 Performance Metrics"])
    
    with tab1:
        st.header("Dataset Overview")
        
        # Load or generate data
        if dataset_name == "elliptic":
            dataset = load_dataset("elliptic")
            if dataset is None:
                st.error("Failed to load Elliptic dataset. Please check if the data is available.")
                return
            snapshots = [dataset[i] for i in range(min(10, len(dataset)))]
        else:
            snapshots = generate_synthetic_data(num_nodes, num_edges, num_timesteps)
        
        # Dataset statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Snapshots", len(snapshots))
        
        with col2:
            st.metric("Nodes per Snapshot", snapshots[0].num_nodes)
        
        with col3:
            st.metric("Edges per Snapshot", snapshots[0].num_edges)
        
        with col4:
            st.metric("Node Features", snapshots[0].num_node_features)
        
        # Class distribution
        st.subheader("Class Distribution")
        all_labels = torch.cat([snapshot.y for snapshot in snapshots])
        class_counts = torch.bincount(all_labels)
        
        fig = px.bar(
            x=[f"Class {i}" for i in range(len(class_counts))],
            y=class_counts.numpy(),
            title="Class Distribution Across All Snapshots"
        )
        fig.update_layout(xaxis_title="Class", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        
        # Temporal evolution
        st.subheader("Temporal Evolution")
        timestep_stats = []
        for i, snapshot in enumerate(snapshots):
            timestep_stats.append({
                "Timestep": i,
                "Nodes": snapshot.num_nodes,
                "Edges": snapshot.num_edges,
                "Avg Degree": (2 * snapshot.num_edges) / snapshot.num_nodes if snapshot.num_nodes > 0 else 0
            })
        
        df_stats = pd.DataFrame(timestep_stats)
        
        fig = px.line(df_stats, x="Timestep", y=["Nodes", "Edges"], 
                     title="Graph Structure Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Graph Visualization")
        
        # Select timestep
        timestep = st.slider("Select Timestep", 0, len(snapshots)-1, 0)
        snapshot = snapshots[timestep]
        
        # Visualize graph structure
        fig = visualize_graph_structure(snapshot)
        st.plotly_chart(fig, use_container_width=True)
        
        # Node details
        st.subheader("Node Details")
        node_id = st.number_input("Node ID", 0, snapshot.num_nodes-1, 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Node {node_id} Features:**")
            features = snapshot.x[node_id].numpy()
            st.write(features)
        
        with col2:
            st.write(f"**Node {node_id} Class:**")
            st.write(f"Class {snapshot.y[node_id].item()}")
            
            # Find neighbors
            neighbors = snapshot.edge_index[1][snapshot.edge_index[0] == node_id]
            st.write(f"**Number of Neighbors:** {len(neighbors)}")
            if len(neighbors) > 0:
                st.write(f"**Neighbors:** {neighbors.tolist()[:10]}{'...' if len(neighbors) > 10 else ''}")
    
    with tab3:
        st.header("Model Analysis")
        
        # Model configuration
        config = {
            "in_channels": snapshots[0].num_node_features,
            "out_channels": len(torch.unique(torch.cat([s.y for s in snapshots]))),
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "dropout": dropout,
            "num_heads": num_heads,
            "time_dim": time_dim
        }
        
        # Create model
        try:
            model = create_model(model_name, config)
            model = model.to(get_device())
            
            # Model information
            st.subheader("Model Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Model:** {model_name}")
                st.write(f"**Parameters:** {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
                st.write(f"**Device:** {get_device()}")
            
            with col2:
                st.write(f"**Input Channels:** {config['in_channels']}")
                st.write(f"**Output Channels:** {config['out_channels']}")
                st.write(f"**Hidden Channels:** {config['hidden_channels']}")
            
            # Model architecture visualization
            st.subheader("Model Architecture")
            
            # Create a simple architecture diagram
            layers = []
            current_channels = config["in_channels"]
            
            for i in range(config["num_layers"]):
                if model_name == "TGAT":
                    layers.append(f"TGAT Layer {i+1}<br>({current_channels} → {config['hidden_channels']})")
                elif model_name == "TGN":
                    layers.append(f"TGN Layer {i+1}<br>({current_channels} → {config['hidden_channels']})")
                elif model_name == "DyRep":
                    layers.append(f"DyRep Layer {i+1}<br>({current_channels} → {config['hidden_channels']})")
                else:
                    layers.append(f"EvolveGCN Layer {i+1}<br>({current_channels} → {config['hidden_channels']})")
                current_channels = config["hidden_channels"]
            
            layers.append(f"Classifier<br>({current_channels} → {config['out_channels']})")
            
            # Create architecture visualization
            fig = go.Figure()
            
            for i, layer in enumerate(layers):
                fig.add_trace(go.Scatter(
                    x=[i], y=[0],
                    mode='markers+text',
                    text=[layer],
                    textposition="middle center",
                    marker=dict(size=100, color='lightblue'),
                    showlegend=False
                ))
            
            fig.update_layout(
                title="Model Architecture",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=200
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating model: {e}")
    
    with tab4:
        st.header("Performance Metrics")
        
        st.info("This section would show model performance metrics after training. "
                "For a complete evaluation, please run the training script.")
        
        # Placeholder metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", "0.85", "0.02")
        
        with col2:
            st.metric("F1 Score", "0.82", "0.01")
        
        with col3:
            st.metric("AUROC", "0.88", "0.03")
        
        with col4:
            st.metric("Avg Precision", "0.84", "0.02")
        
        # Performance comparison
        st.subheader("Model Comparison")
        
        models = ["EvolveGCN", "TGAT", "TGN", "DyRep", "Ensemble"]
        metrics = {
            "Accuracy": [0.82, 0.85, 0.83, 0.81, 0.87],
            "F1 Score": [0.79, 0.82, 0.80, 0.78, 0.84],
            "AUROC": [0.85, 0.88, 0.86, 0.84, 0.90],
            "Avg Precision": [0.81, 0.84, 0.82, 0.80, 0.86]
        }
        
        df_comparison = pd.DataFrame(metrics, index=models)
        
        fig = px.bar(
            df_comparison,
            title="Model Performance Comparison",
            barmode='group'
        )
        fig.update_layout(xaxis_title="Models", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
